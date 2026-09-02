"""Leakage-free state, route, and intervention geometry for response Crystals."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .branches import (
    InterventionOperation,
    InterventionPlan,
    apply_intervention,
)
from .causal_graph import CausalEdge
from .contracts import ForecastInputs
from .identity import require_sha256, sha256_bytes, sha256_json
from .transport import route_transport_w1

FEATURE_ABI = "o1fm.intervention-response-features/v1"
MAX_NAMES = 4096
MAX_NAME_BYTES = 1024
MAX_HORIZON = 4096
MAX_SEASONAL_BASES = 32
MAX_ROUTE_EDGES = 100_000
MAX_FEATURE_DIMENSION = 4_000_000
MAX_DESIGN_DIMENSION = 1024
MAX_RAW_ABSOLUTE = 1.0e40
_LOG_SCALE = math.log1p(MAX_RAW_ABSOLUTE)

STATE_BASE_FEATURES = (
    "last_level",
    "mean_level",
    "slope",
    "volatility",
    "observed_fraction",
    "last_observed_age",
    "latest_observed",
)
STATE_SEASONAL_FEATURES = (
    "lag_level",
    "last_minus_lag",
    "lag_observed",
    "phase_sin",
    "phase_cos",
)
ROUTE_TARGET_FEATURES = (
    "incoming_count",
    "positive_score_mass",
    "negative_score_mass",
    "abs_score_weighted_lag",
    "intervention_score_mass",
)
ROUTE_CHANNEL_TARGET_FEATURES = (
    "signed_score_mass",
    "absolute_score_mass",
    "abs_score_weighted_lag",
    "log_observation_mass",
)
OPERATOR_CELL_FEATURES = (
    "active",
    "operation_set",
    "operation_shift",
    "operation_scale",
    "operand",
    "applied_delta",
    "absolute_delta",
    "sign_positive",
    "sign_negative",
    "sign_zero",
    "base_observed",
    "revealed",
    "time",
    "run_start",
    "run_end",
    "run_position",
    "run_duration",
    "pulse",
    "ramp",
    "operand_slope",
)


class ResponseFeatureError(ValueError):
    """Inputs cannot be represented by one exact response feature schema."""


def _name(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > MAX_NAME_BYTES
    ):
        raise ResponseFeatureError(f"{field} must be canonical bounded text")
    return value


def _names(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ResponseFeatureError(f"{field} must be a sequence")
    selected = tuple(_name(item, field=field) for item in value)
    if not selected or len(selected) > MAX_NAMES or len(set(selected)) != len(selected):
        raise ResponseFeatureError(f"{field} must be non-empty, bounded, and unique")
    return selected


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= maximum
    ):
        raise ResponseFeatureError(f"{field} must be a positive bounded integer")
    return int(value)


def _finite(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ResponseFeatureError(f"{field} must be finite")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ResponseFeatureError(f"{field} lies outside its finite bound")
    return 0.0 if result == 0.0 else result


def _bounded(value: float) -> float:
    if not math.isfinite(value) or abs(value) > MAX_RAW_ABSOLUTE:
        raise ResponseFeatureError("raw feature magnitude exceeds its bound")
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(value)) / _LOG_SCALE, value)


def _readonly(
    value: object,
    *,
    shape: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    try:
        result = np.array(value, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResponseFeatureError("feature array must be numeric") from exc
    if (
        (shape is not None and result.shape != shape)
        or result.size < 1
        or result.size > MAX_FEATURE_DIMENSION
        or not bool(np.isfinite(result).all())
        or bool((np.abs(result) > 1.0 + 1e-12).any())
    ):
        raise ResponseFeatureError("feature array violates its shape or value bound")
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _array_record(value: NDArray[np.generic]) -> dict[str, object]:
    dtype = value.dtype.newbyteorder("<")
    array = np.ascontiguousarray(value, dtype=dtype)
    return {
        "data_sha256": sha256_bytes(array.tobytes(order="C")),
        "dtype": array.dtype.str,
        "shape": [int(axis) for axis in array.shape],
    }


@dataclass(frozen=True, slots=True)
class ResponseFeatureSchema:
    """Exact ABI for one response feature space and D-optimal projection."""

    target_names: Sequence[str]
    known_covariate_names: Sequence[str]
    horizon: int
    graph_topology_sha256: str
    seasonal_periods: Sequence[int] = (24,)
    state_window: int = 64
    mask_tail: int = 8
    design_dimension: int = 256
    projection_scope_sha256: str | None = None

    def __post_init__(self) -> None:
        targets = _names(self.target_names, field="target_names")
        known = _names(self.known_covariate_names, field="known_covariate_names")
        if set(targets) & set(known):
            raise ResponseFeatureError(
                "target and known-covariate names must be disjoint"
            )
        horizon = _positive_int(self.horizon, field="horizon", maximum=MAX_HORIZON)
        topology = require_sha256(
            self.graph_topology_sha256,
            field="graph_topology_sha256",
        )
        if isinstance(self.seasonal_periods, (str, bytes)) or not isinstance(
            self.seasonal_periods, Sequence
        ):
            raise ResponseFeatureError("seasonal_periods must be a sequence")
        periods = tuple(
            _positive_int(item, field="seasonal period", maximum=MAX_HORIZON * 1024)
            for item in self.seasonal_periods
        )
        if (
            not periods
            or len(periods) > MAX_SEASONAL_BASES
            or tuple(sorted(set(periods))) != periods
        ):
            raise ResponseFeatureError(
                "seasonal_periods must be non-empty, unique, and ordered"
            )
        state_window = _positive_int(
            self.state_window,
            field="state_window",
            maximum=MAX_HORIZON * 1024,
        )
        mask_tail = _positive_int(self.mask_tail, field="mask_tail", maximum=1024)
        design = _positive_int(
            self.design_dimension,
            field="design_dimension",
            maximum=MAX_DESIGN_DIMENSION,
        )
        projection_scope = (
            None
            if self.projection_scope_sha256 is None
            else require_sha256(
                self.projection_scope_sha256,
                field="projection_scope_sha256",
            )
        )
        object.__setattr__(self, "target_names", targets)
        object.__setattr__(self, "known_covariate_names", known)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "graph_topology_sha256", topology)
        object.__setattr__(self, "seasonal_periods", periods)
        object.__setattr__(self, "state_window", state_window)
        object.__setattr__(self, "mask_tail", mask_tail)
        object.__setattr__(self, "design_dimension", design)
        object.__setattr__(self, "projection_scope_sha256", projection_scope)
        if self.feature_dimension > MAX_FEATURE_DIMENSION:
            raise ResponseFeatureError("response feature layout exceeds its bound")

    @property
    def target_count(self) -> int:
        return len(self.target_names)

    @property
    def known_covariate_count(self) -> int:
        return len(self.known_covariate_names)

    @property
    def state_features_per_target(self) -> int:
        return (
            len(STATE_BASE_FEATURES)
            + self.mask_tail
            + len(self.seasonal_periods) * len(STATE_SEASONAL_FEATURES)
        )

    @property
    def state_dimension(self) -> int:
        return self.target_count * self.state_features_per_target

    @property
    def route_dimension(self) -> int:
        return (
            1
            + self.target_count
            + self.target_count * len(ROUTE_TARGET_FEATURES)
            + self.known_covariate_count
            * self.target_count
            * len(ROUTE_CHANNEL_TARGET_FEATURES)
        )

    @property
    def operator_dimension(self) -> int:
        return self.known_covariate_count * self.horizon * len(OPERATOR_CELL_FEATURES)

    @property
    def feature_dimension(self) -> int:
        return 1 + self.state_dimension + self.route_dimension + self.operator_dimension

    @property
    def target_schema_sha256(self) -> str:
        return sha256_json(
            {
                "ordered_targets": list(self.target_names),
                "schema": "o1fm.ordered-target-schema/v1",
            }
        )

    @property
    def known_covariate_schema_sha256(self) -> str:
        return sha256_json(
            {
                "ordered_known_covariates": list(self.known_covariate_names),
                "schema": "o1fm.ordered-known-covariate-schema/v1",
            }
        )

    def _layout_record(self) -> dict[str, object]:
        return {
            "abi": FEATURE_ABI,
            "design_dimension": self.design_dimension,
            "horizon": self.horizon,
            "known_covariate_names": list(self.known_covariate_names),
            "layout": {
                "bias": ["bias"],
                "operator_cell": list(OPERATOR_CELL_FEATURES),
                "route_channel_target": list(ROUTE_CHANNEL_TARGET_FEATURES),
                "route_target": list(ROUTE_TARGET_FEATURES),
                "state_base": list(STATE_BASE_FEATURES),
                "state_mask_tail": self.mask_tail,
                "state_seasonal": list(STATE_SEASONAL_FEATURES),
            },
            "seasonal_basis": {
                "periods": list(self.seasonal_periods),
                "phase": "timestamp-ns/median-positive-step/v1",
            },
            "state_window": self.state_window,
            "target_names": list(self.target_names),
        }

    @property
    def layout_identity_sha256(self) -> str:
        """Topology-independent ABI for parent response feature spaces."""

        return sha256_json(
            {
                **self._layout_record(),
                "schema": "o1fm.intervention-response-feature-layout/v1",
            }
        )

    def to_record(self) -> dict[str, object]:
        return {
            **self._layout_record(),
            "design": {
                "dimension": self.design_dimension,
                "projection": "schema-seeded-signed-hash-tanh/v1",
                "projection_scope_sha256": self.projection_scope_sha256,
            },
            "feature_dimension": self.feature_dimension,
            "graph_topology_sha256": self.graph_topology_sha256,
            "known_covariate_schema_sha256": self.known_covariate_schema_sha256,
            "layout_identity_sha256": self.layout_identity_sha256,
            "target_schema_sha256": self.target_schema_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.to_record())

    @property
    def sha256(self) -> str:
        return self.identity_sha256

    @property
    def resolved_projection_scope_sha256(self) -> str:
        """Explicit parent scope, or this exact topology-bound schema."""

        return (
            self.identity_sha256
            if self.projection_scope_sha256 is None
            else self.projection_scope_sha256
        )

    def validate_inputs(self, inputs: ForecastInputs) -> None:
        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        if (
            inputs.horizon != self.horizon
            or inputs.target_dim != self.target_count
            or inputs.known_covariate_dim != self.known_covariate_count
        ):
            raise ResponseFeatureError("inputs differ from the response feature schema")


@dataclass(frozen=True, slots=True)
class ResponseRouteContext:
    """Pre-outcome route evidence bound to one exact graph topology."""

    graph_topology_sha256: str
    edges: Sequence[CausalEdge] = ()
    per_target_w1: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        topology = require_sha256(
            self.graph_topology_sha256,
            field="graph_topology_sha256",
        )
        if isinstance(self.edges, (str, bytes)) or not isinstance(self.edges, Sequence):
            raise ResponseFeatureError("edges must be a sequence")
        edges = tuple(self.edges)
        if len(edges) > MAX_ROUTE_EDGES or not all(
            isinstance(edge, CausalEdge) for edge in edges
        ):
            raise ResponseFeatureError("edges must contain bounded CausalEdge values")
        identities = tuple(
            (
                edge.source,
                edge.target,
                edge.lag,
                edge.kind,
                edge.revision_sha256,
            )
            for edge in edges
        )
        if len(set(identities)) != len(identities):
            raise ResponseFeatureError("route edges must be unique")
        edges = tuple(
            sorted(
                edges,
                key=lambda edge: (
                    edge.source,
                    edge.target,
                    edge.lag,
                    edge.kind,
                    edge.revision_sha256,
                ),
            )
        )
        if self.per_target_w1 is None:
            w1 = None
        else:
            if not isinstance(self.per_target_w1, Mapping):
                raise ResponseFeatureError("per_target_w1 must be a mapping")
            w1 = MappingProxyType(
                {
                    _name(name, field="route target"): _finite(
                        value,
                        field="route W1",
                        minimum=0.0,
                        maximum=MAX_RAW_ABSOLUTE,
                    )
                    for name, value in sorted(self.per_target_w1.items())
                }
            )
        object.__setattr__(self, "graph_topology_sha256", topology)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "per_target_w1", w1)

    @property
    def edge_set_sha256(self) -> str:
        return sha256_json(
            {
                "edges": [edge.to_record() for edge in self.edges],
                "schema": "o1fm.response-route-edge-set/v1",
            }
        )


@dataclass(frozen=True, slots=True)
class InterventionResponseFeatures:
    """Immutable full feature tensor and compact pre-outcome design vector."""

    schema: ResponseFeatureSchema
    plan_identity_sha256: str
    input_context_sha256: str
    route_edge_set_sha256: str
    state: object
    route: object
    operator: object
    vector: object
    design_vector: object

    def __post_init__(self) -> None:
        if not isinstance(self.schema, ResponseFeatureSchema):
            raise TypeError("schema must be ResponseFeatureSchema")
        for field in (
            "plan_identity_sha256",
            "input_context_sha256",
            "route_edge_set_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        state = _readonly(self.state, shape=(self.schema.state_dimension,))
        route = _readonly(self.route, shape=(self.schema.route_dimension,))
        operator = _readonly(
            self.operator,
            shape=(
                self.schema.known_covariate_count,
                self.schema.horizon,
                len(OPERATOR_CELL_FEATURES),
            ),
        )
        vector = _readonly(self.vector, shape=(self.schema.feature_dimension,))
        expected = np.concatenate(
            (
                np.ones(1, dtype=np.float64),
                state,
                route,
                operator.reshape(-1),
            )
        )
        if not np.array_equal(vector, expected):
            raise ResponseFeatureError("feature vector does not match its components")
        design = _readonly(
            self.design_vector,
            shape=(self.schema.design_dimension,),
        )
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "design_vector", design)

    @property
    def pre_outcome_design(self) -> NDArray[np.float64]:
        return self.design_vector

    @property
    def identity_sha256(self) -> str:
        return sha256_json(
            {
                "design": _array_record(self.design_vector),
                "input_context_sha256": self.input_context_sha256,
                "operator": _array_record(self.operator),
                "plan_identity_sha256": self.plan_identity_sha256,
                "route": _array_record(self.route),
                "route_edge_set_sha256": self.route_edge_set_sha256,
                "schema": "o1fm.intervention-response-feature-instance/v1",
                "schema_sha256": self.schema.identity_sha256,
                "state": _array_record(self.state),
                "vector": _array_record(self.vector),
            }
        )

    @property
    def sha256(self) -> str:
        return self.identity_sha256


def _context_sha256(inputs: ForecastInputs) -> str:
    return sha256_json(
        {
            "horizon": inputs.horizon,
            "known_covariate_mask": _array_record(inputs.known_covariate_mask),
            "known_covariates": _array_record(inputs.known_covariates),
            "past_covariate_mask": _array_record(inputs.past_covariate_mask),
            "past_covariates": _array_record(inputs.past_covariates),
            "past_target": _array_record(inputs.past_target),
            "past_target_mask": _array_record(inputs.past_target_mask),
            "schema": "o1fm.response-feature-input-context/v1",
            "series_id": inputs.series_id,
            "timestamps_ns": _array_record(
                inputs.timestamps.astype(np.int64, copy=False)
            ),
        }
    )


def _column_state(
    values: NDArray[np.float32],
    mask: NDArray[np.bool_],
    schema: ResponseFeatureSchema,
    phases: Mapping[int, float],
) -> list[float]:
    length = int(values.size)
    start = max(0, length - schema.state_window)
    selected_values = np.asarray(values[start:], dtype=np.float64)
    selected_mask = np.asarray(mask[start:], dtype=np.bool_)
    indices = np.flatnonzero(selected_mask)
    if indices.size:
        observed = selected_values[indices]
        last_index = int(indices[-1])
        last = float(observed[-1])
        mean = float(observed.mean())
        if indices.size > 1:
            x = indices.astype(np.float64)
            centered = x - x.mean()
            denominator = float(centered @ centered)
            slope = (
                0.0
                if denominator == 0.0
                else float(centered @ (observed - observed.mean())) / denominator
            )
        else:
            slope = 0.0
        pair_mask = selected_mask[1:] & selected_mask[:-1]
        differences = np.diff(selected_values)[pair_mask]
        volatility = 0.0 if differences.size == 0 else float(differences.std())
        fraction = float(indices.size / max(1, selected_mask.size))
        age = float((selected_mask.size - 1 - last_index) / max(1, selected_mask.size))
        latest = float(selected_mask[-1])
    else:
        last = mean = slope = volatility = 0.0
        fraction = 0.0
        age = 1.0
        latest = 0.0
    result = [
        _bounded(last),
        _bounded(mean),
        _bounded(slope),
        _bounded(volatility),
        fraction,
        age,
        latest,
    ]
    tail = np.zeros(schema.mask_tail, dtype=np.float64)
    available = min(schema.mask_tail, length)
    tail[-available:] = np.asarray(mask[-available:], dtype=np.float64)
    result.extend(float(value) for value in tail)
    for period in schema.seasonal_periods:
        lag_index = length - 1 - period
        lag_observed = lag_index >= 0 and bool(mask[lag_index])
        lag = float(values[lag_index]) if lag_observed else 0.0
        phase = phases[period]
        result.extend(
            (
                _bounded(lag),
                _bounded(last - lag) if lag_observed and indices.size else 0.0,
                float(lag_observed),
                math.sin(phase),
                math.cos(phase),
            )
        )
    return result


def _state_vector(
    inputs: ForecastInputs,
    schema: ResponseFeatureSchema,
) -> NDArray[np.float64]:
    ticks = inputs.timestamps.astype(np.int64, copy=False)
    if ticks.size > 1:
        step = int(np.median(np.diff(ticks)))
        if step <= 0:
            raise ResponseFeatureError("timestamps cannot define a seasonal phase")
    else:  # ForecastInputs currently always has history plus future rows.
        step = 1
    origin_step = int(inputs.future_timestamps[0].astype(np.int64)) // step
    phases = {
        period: 2.0 * math.pi * float(origin_step % period) / float(period)
        for period in schema.seasonal_periods
    }
    values: list[float] = []
    for target in range(schema.target_count):
        values.extend(
            _column_state(
                inputs.past_target[:, target],
                inputs.past_target_mask[:, target],
                schema,
                phases,
            )
        )
    return _readonly(values, shape=(schema.state_dimension,))


def _route_vector(
    context: ResponseRouteContext,
    schema: ResponseFeatureSchema,
) -> NDArray[np.float64]:
    if context.graph_topology_sha256 != schema.graph_topology_sha256:
        raise ResponseFeatureError("route context belongs to another graph topology")
    if context.per_target_w1 is None:
        result = route_transport_w1(context.edges, schema.target_names)
        per_target = dict(result.per_target_w1)
        mean_w1 = result.mean_w1
    else:
        per_target = dict(context.per_target_w1)
        if set(per_target) != set(schema.target_names):
            raise ResponseFeatureError(
                "per_target_w1 must cover the ordered target schema exactly"
            )
        mean_w1 = math.fsum(per_target[name] for name in schema.target_names) / len(
            schema.target_names
        )
    rows: list[float] = [_bounded(mean_w1)]
    rows.extend(_bounded(per_target[name]) for name in schema.target_names)
    by_target = {
        name: tuple(edge for edge in context.edges if edge.target == name)
        for name in schema.target_names
    }

    def signed_score(edge: CausalEdge) -> float:
        return -edge.score if edge.covariance < 0.0 else edge.score

    for target in schema.target_names:
        edges = by_target[target]
        count = float(len(edges) / max(1, MAX_ROUTE_EDGES))
        positive = math.fsum(max(0.0, signed_score(edge)) for edge in edges)
        negative = math.fsum(min(0.0, signed_score(edge)) for edge in edges)
        absolute = math.fsum(abs(edge.score) for edge in edges)
        lag = (
            0.0
            if absolute == 0.0
            else math.fsum(abs(edge.score) * edge.lag for edge in edges) / absolute
        )
        intervention = math.fsum(
            abs(edge.score) for edge in edges if edge.kind == "intervention"
        )
        rows.extend(
            (
                count,
                _bounded(positive),
                _bounded(negative),
                _bounded(lag),
                _bounded(intervention),
            )
        )
    for source in schema.known_covariate_names:
        for target in schema.target_names:
            edges = tuple(
                edge
                for edge in context.edges
                if edge.source == source and edge.target == target
            )
            signed = math.fsum(signed_score(edge) for edge in edges)
            absolute = math.fsum(abs(edge.score) for edge in edges)
            lag = (
                0.0
                if absolute == 0.0
                else math.fsum(abs(edge.score) * edge.lag for edge in edges) / absolute
            )
            observations = math.fsum(edge.observations for edge in edges)
            rows.extend(
                (
                    _bounded(signed),
                    _bounded(absolute),
                    _bounded(lag),
                    _bounded(math.log1p(observations)),
                )
            )
    return _readonly(rows, shape=(schema.route_dimension,))


def _runs(active: NDArray[np.bool_]) -> tuple[tuple[int, ...], ...]:
    selected = np.flatnonzero(active)
    if selected.size == 0:
        return ()
    groups: list[list[int]] = [[int(selected[0])]]
    for raw in selected[1:]:
        index = int(raw)
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _operator_tensor(
    inputs: ForecastInputs,
    plan: InterventionPlan,
    schema: ResponseFeatureSchema,
) -> NDArray[np.float64]:
    if not isinstance(plan, InterventionPlan):
        raise TypeError("plan must be InterventionPlan")
    edited = apply_intervention(inputs, plan)
    feature_count = len(OPERATOR_CELL_FEATURES)
    tensor = np.zeros(
        (schema.known_covariate_count, schema.horizon, feature_count),
        dtype=np.float64,
        order="C",
    )
    future_ticks = inputs.future_timestamps.astype(np.int64, copy=False)
    row_by_tick = {int(tick): row for row, tick in enumerate(future_ticks)}
    history = inputs.history_length
    operation_offset = {
        InterventionOperation.SET: 1,
        InterventionOperation.SHIFT: 2,
        InterventionOperation.SCALE: 3,
    }
    for spec in plan.interventions:
        local_to_future = tuple(
            row_by_tick.get(int(tick), -1)
            for tick in spec.timestamps.astype(np.int64, copy=False)
        )
        if any(index < 0 for index in local_to_future):
            raise ResponseFeatureError(
                "intervention timestamp lies outside the feature horizon"
            )
        for local_column, channel in enumerate(spec.covariate_indices):
            if channel >= schema.known_covariate_count:
                raise ResponseFeatureError(
                    "intervention channel lies outside the known-covariate schema"
                )
            active = np.zeros(schema.horizon, dtype=np.bool_)
            operand = np.zeros(schema.horizon, dtype=np.float64)
            reveal = np.zeros(schema.horizon, dtype=np.bool_)
            for local_row, future_row in enumerate(local_to_future):
                if not bool(spec.edit_mask[local_row, local_column]):
                    continue
                active[future_row] = True
                operand[future_row] = float(spec.values[local_row, local_column])
                reveal[future_row] = bool(
                    spec.observed_mask is not None
                    and spec.observed_mask[local_row, local_column]
                )
            for run in _runs(active):
                duration = len(run)
                pulse = float(duration == 1)
                run_values = operand[list(run)]
                if duration > 1:
                    x = np.arange(duration, dtype=np.float64)
                    centered = x - x.mean()
                    denominator = float(centered @ centered)
                    slope = (
                        0.0
                        if denominator == 0.0
                        else float(centered @ (run_values - run_values.mean()))
                        / denominator
                    )
                else:
                    slope = 0.0
                ramp = float(duration > 1 and abs(slope) > 1e-12)
                for position, future_row in enumerate(run):
                    base_row = history + future_row
                    before = float(inputs.known_covariates[base_row, channel])
                    after = float(edited.known_covariates[base_row, channel])
                    delta = after - before
                    sign_positive = float(delta > 0.0)
                    sign_negative = float(delta < 0.0)
                    sign_zero = float(delta == 0.0)
                    cell = tensor[channel, future_row]
                    cell[0] = 1.0
                    cell[operation_offset[spec.operation]] = 1.0
                    cell[4] = _bounded(operand[future_row])
                    cell[5] = _bounded(delta)
                    cell[6] = _bounded(abs(delta))
                    cell[7] = sign_positive
                    cell[8] = sign_negative
                    cell[9] = sign_zero
                    cell[10] = float(inputs.known_covariate_mask[base_row, channel])
                    cell[11] = float(reveal[future_row])
                    cell[12] = float(future_row / max(1, schema.horizon - 1))
                    cell[13] = float(position == 0)
                    cell[14] = float(position == duration - 1)
                    cell[15] = float(position / max(1, duration - 1))
                    cell[16] = float(duration / schema.horizon)
                    cell[17] = pulse
                    cell[18] = ramp
                    cell[19] = _bounded(slope)
    return _readonly(
        tensor,
        shape=(
            schema.known_covariate_count,
            schema.horizon,
            len(OPERATOR_CELL_FEATURES),
        ),
    )


def _project_design(
    vector: NDArray[np.float64],
    schema: ResponseFeatureSchema,
) -> NDArray[np.float64]:
    result = np.zeros(schema.design_dimension, dtype=np.float64)
    counts = np.zeros(schema.design_dimension, dtype=np.int64)
    namespace = bytes.fromhex(schema.resolved_projection_scope_sha256)
    for index, value in enumerate(vector):
        digest = hashlib.sha256(namespace + index.to_bytes(8, "big")).digest()
        bucket = int.from_bytes(digest[:8], "big") % schema.design_dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        result[bucket] += sign * float(value)
        counts[bucket] += 1
    selected = counts > 0
    result[selected] /= np.sqrt(counts[selected].astype(np.float64))
    result = np.tanh(result)
    return _readonly(result, shape=(schema.design_dimension,))


class InterventionResponseFeatureEncoder:
    """Encode any legal intervention without exposing a future-target channel."""

    def __init__(self, schema: ResponseFeatureSchema) -> None:
        if not isinstance(schema, ResponseFeatureSchema):
            raise TypeError("schema must be ResponseFeatureSchema")
        self.schema = schema

    def encode(
        self,
        inputs: ForecastInputs,
        plan: InterventionPlan,
        route: ResponseRouteContext,
    ) -> InterventionResponseFeatures:
        self.schema.validate_inputs(inputs)
        if not isinstance(route, ResponseRouteContext):
            raise TypeError("route must be ResponseRouteContext")
        state = _state_vector(inputs, self.schema)
        route_vector = _route_vector(route, self.schema)
        operator = _operator_tensor(inputs, plan, self.schema)
        vector = _readonly(
            np.concatenate(
                (
                    np.ones(1, dtype=np.float64),
                    state,
                    route_vector,
                    operator.reshape(-1),
                )
            ),
            shape=(self.schema.feature_dimension,),
        )
        design = _project_design(vector, self.schema)
        return InterventionResponseFeatures(
            schema=self.schema,
            plan_identity_sha256=plan.identity,
            input_context_sha256=_context_sha256(inputs),
            route_edge_set_sha256=route.edge_set_sha256,
            state=state,
            route=route_vector,
            operator=operator,
            vector=vector,
            design_vector=design,
        )

    def design_vector(
        self,
        inputs: ForecastInputs,
        plan: InterventionPlan,
        route: ResponseRouteContext,
    ) -> NDArray[np.float64]:
        return self.encode(inputs, plan, route).design_vector


def encode_intervention_response_features(
    schema: ResponseFeatureSchema,
    inputs: ForecastInputs,
    plan: InterventionPlan,
    route: ResponseRouteContext,
) -> InterventionResponseFeatures:
    return InterventionResponseFeatureEncoder(schema).encode(inputs, plan, route)


__all__ = [
    "FEATURE_ABI",
    "MAX_DESIGN_DIMENSION",
    "OPERATOR_CELL_FEATURES",
    "ROUTE_CHANNEL_TARGET_FEATURES",
    "ROUTE_TARGET_FEATURES",
    "STATE_BASE_FEATURES",
    "STATE_SEASONAL_FEATURES",
    "InterventionResponseFeatureEncoder",
    "InterventionResponseFeatures",
    "ResponseFeatureError",
    "ResponseFeatureSchema",
    "ResponseRouteContext",
    "encode_intervention_response_features",
]
