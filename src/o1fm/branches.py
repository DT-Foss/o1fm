"""Explicit factual, interventional, and counterfactual forecast branches.

The contracts in this module keep scenario construction separate from realized
future targets.  An intervention can edit only known-future covariate cells;
causal language is admitted only when the forecast execution returns matching
``CausalEdge(kind="intervention")`` evidence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .atlas import AtlasReceipt
from .causal_graph import CausalEdge
from .contracts import ForecastInputs, ForecastPrediction
from .identity import canonical_json_bytes, require_sha256, sha256_bytes, sha256_json

_MAX_TEXT_BYTES = 1024
_MAX_INTERVENTION_CELLS = 1_000_000
_FLOAT32_LIMIT = float(np.finfo(np.float32).max)


def _text(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_TEXT_BYTES
    ):
        raise ValueError(f"{field_name} must be canonical non-empty text")
    canonical_json_bytes(value)
    return value


def _texts(values: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    return tuple(_text(value, field_name=field_name) for value in values)


def _finite(value: object, *, field_name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field_name} must be finite")
    selected = float(value)
    if (
        not math.isfinite(selected)
        or abs(selected) > _FLOAT32_LIMIT
        or (nonnegative and selected < 0.0)
    ):
        raise ValueError(f"{field_name} must be finite and bounded")
    return 0.0 if selected == 0.0 else selected


def _readonly_float32(
    value: object,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> NDArray[np.float32]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{field_name} cannot be complex")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(value, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if result.shape != shape or not bool(np.isfinite(result).all()):
        raise ValueError(f"{field_name} must be a finite matrix with shape {shape}")
    result[result == 0.0] = np.float32(0.0)
    result.setflags(write=False)
    return result


def _readonly_bool(
    value: object | None,
    *,
    shape: tuple[int, ...],
    field_name: str,
    default: bool,
) -> NDArray[np.bool_]:
    if value is None:
        result = np.full(shape, default, dtype=np.bool_, order="C")
    else:
        raw = np.asarray(value)
        if raw.dtype != np.dtype(np.bool_) or raw.shape != shape:
            raise ValueError(
                f"{field_name} must be a boolean matrix with shape {shape}"
            )
        result = np.array(raw, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _timestamps(value: object, *, field_name: str) -> NDArray[np.datetime64]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a timestamp sequence")
    try:
        raw = np.asarray(value)
        result = np.array(value, dtype="datetime64[ns]", order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must contain valid timestamps") from exc
    if raw.ndim != 1 or result.ndim != 1 or result.size == 0:
        raise ValueError(f"{field_name} must be a non-empty timestamp vector")
    if bool(np.isnat(result).any()):
        raise ValueError(f"{field_name} cannot contain missing timestamps")
    ticks = result.astype(np.int64, copy=False)
    if result.size > 1 and bool((np.diff(ticks) <= 0).any()):
        raise ValueError(f"{field_name} must be strictly increasing and unique")
    result.setflags(write=False)
    return result


def _indices(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("covariate_indices must be a sequence")
    result: list[int] = []
    for child in value:
        if (
            isinstance(child, bool)
            or not isinstance(child, (int, np.integer))
            or int(child) < 0
        ):
            raise ValueError("covariate indices must be non-negative integers")
        result.append(int(child))
    if not result or tuple(sorted(set(result))) != tuple(result):
        raise ValueError("covariate indices must be sorted and unique")
    return tuple(result)


def _bounds(
    value: object | None,
    *,
    count: int,
    field_name: str,
    default: float,
) -> tuple[float, ...]:
    if value is None:
        return (default,) * count
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    if len(value) != count:
        raise ValueError(f"{field_name} must contain one bound per covariate")
    return tuple(_finite(child, field_name=field_name) for child in value)


def _array_digest(value: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(value)
    return sha256_bytes(contiguous.tobytes(order="C"))


def _prediction_record(prediction: ForecastPrediction) -> dict[str, object]:
    return {
        "point": {
            "dtype": "float32",
            "sha256": _array_digest(prediction.point),
            "shape": list(prediction.point.shape),
        },
        "quantile_levels": (
            None
            if prediction.quantile_levels is None
            else {
                "dtype": "float32",
                "sha256": _array_digest(prediction.quantile_levels),
                "shape": list(prediction.quantile_levels.shape),
            }
        ),
        "quantiles": (
            None
            if prediction.quantiles is None
            else {
                "dtype": "float32",
                "sha256": _array_digest(prediction.quantiles),
                "shape": list(prediction.quantiles.shape),
            }
        ),
        "series_id": prediction.series_id,
        "timestamps_ns_sha256": _array_digest(
            prediction.timestamps.astype(np.int64, copy=False)
        ),
    }


class InterventionOperation(StrEnum):
    """Allowed operations on known-future covariate cells."""

    SET = "set"
    SHIFT = "shift"
    SCALE = "scale"


class IdentificationStatus(StrEnum):
    """How a forecast branch may be interpreted."""

    FACTUAL = "factual"
    OBSERVED_POLICY_INTERVENTION = "observed_policy_intervention"
    GRAPH_SUPPORTED_INTERVENTION = "graph_supported_intervention"
    COUNTERFACTUAL_HYPOTHESIS = "counterfactual_hypothesis"


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    """One bounded edit over exact known-future covariate coordinates.

    ``edit_mask`` selects edited cells.  ``observed_mask`` is optional positive
    evidence that selected values are known; it can reveal selected cells but
    can never hide an already observed cell.  A scenario therefore cannot
    mutate any input mask unless it explicitly carries observed-future values.
    """

    name: str
    operation: InterventionOperation | str
    timestamps: object
    covariate_indices: Sequence[int]
    values: object
    edit_mask: object | None = None
    observed_mask: object | None = None
    lower_bounds: Sequence[float] | None = None
    upper_bounds: Sequence[float] | None = None

    def __post_init__(self) -> None:
        name = _text(self.name, field_name="intervention name")
        try:
            operation = InterventionOperation(self.operation)
        except ValueError as exc:
            raise ValueError("operation must be set, shift, or scale") from exc
        timestamps = _timestamps(self.timestamps, field_name="intervention timestamps")
        indices = _indices(self.covariate_indices)
        shape = (int(timestamps.size), len(indices))
        if math.prod(shape) > _MAX_INTERVENTION_CELLS:
            raise ValueError("intervention exceeds its cell bound")
        values = _readonly_float32(self.values, shape=shape, field_name="values")
        edit_mask = _readonly_bool(
            self.edit_mask,
            shape=shape,
            field_name="edit_mask",
            default=True,
        )
        if not bool(edit_mask.any()):
            raise ValueError("intervention must edit at least one cell")
        if self.observed_mask is None:
            observed_mask = None
        else:
            observed_mask = _readonly_bool(
                self.observed_mask,
                shape=shape,
                field_name="observed_mask",
                default=False,
            )
            if bool((observed_mask & ~edit_mask).any()):
                raise ValueError("observed_mask may reveal only edited cells")
        lower = _bounds(
            self.lower_bounds,
            count=len(indices),
            field_name="lower_bounds",
            default=-_FLOAT32_LIMIT,
        )
        upper = _bounds(
            self.upper_bounds,
            count=len(indices),
            field_name="upper_bounds",
            default=_FLOAT32_LIMIT,
        )
        if any(left > right for left, right in zip(lower, upper, strict=True)):
            raise ValueError("lower_bounds cannot exceed upper_bounds")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "covariate_indices", indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "edit_mask", edit_mask)
        object.__setattr__(self, "observed_mask", observed_mask)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)

    def to_record(self) -> dict[str, object]:
        timestamps = self.timestamps.astype(np.int64, copy=False)
        return {
            "covariate_indices": list(self.covariate_indices),
            "edit_mask": self.edit_mask.tolist(),
            "lower_bounds": [value.hex() for value in self.lower_bounds],
            "name": self.name,
            "observed_mask": (
                None if self.observed_mask is None else self.observed_mask.tolist()
            ),
            "operation": self.operation.value,
            "schema": "o1fm.intervention-spec/v1",
            "timestamps_ns": [int(value) for value in timestamps],
            "upper_bounds": [value.hex() for value in self.upper_bounds],
            "values_float32_hex": [
                [float(value).hex() for value in row] for row in self.values
            ],
        }

    @property
    def identity(self) -> str:
        return sha256_json(self.to_record())

    @property
    def canonical_identity(self) -> str:
        return self.identity

    @property
    def edited_covariates(self) -> tuple[int, ...]:
        active = np.any(self.edit_mask, axis=0)
        return tuple(
            index
            for index, selected in zip(self.covariate_indices, active, strict=True)
            if bool(selected)
        )


@dataclass(frozen=True, slots=True)
class InterventionPlan:
    """A canonical, order-independent collection of non-overlapping edits."""

    name: str
    interventions: Sequence[InterventionSpec]
    observed_policy: bool = False
    assumptions: Sequence[str] = ()

    def __post_init__(self) -> None:
        name = _text(self.name, field_name="plan name")
        if isinstance(self.interventions, (str, bytes)) or not isinstance(
            self.interventions, Sequence
        ):
            raise TypeError("interventions must be a sequence")
        specs = tuple(self.interventions)
        if not specs or not all(isinstance(spec, InterventionSpec) for spec in specs):
            raise TypeError("plan requires InterventionSpec instances")
        if not isinstance(self.observed_policy, bool):
            raise TypeError("observed_policy must be boolean")
        assumptions = _texts(self.assumptions, field_name="assumption")
        identities = [spec.identity for spec in specs]
        if len(set(identities)) != len(identities):
            raise ValueError("plan cannot repeat an intervention")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("intervention names must be unique within a plan")
        occupied: set[tuple[int, int]] = set()
        for spec in specs:
            ticks = spec.timestamps.astype(np.int64, copy=False)
            for row, tick in enumerate(ticks):
                for column, covariate in enumerate(spec.covariate_indices):
                    cell = (int(tick), covariate)
                    if bool(spec.edit_mask[row, column]) and cell in occupied:
                        raise ValueError("plan contains overlapping intervention cells")
                    if bool(spec.edit_mask[row, column]):
                        occupied.add(cell)
        specs = tuple(sorted(specs, key=lambda item: (item.name, item.identity)))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "interventions", specs)
        object.__setattr__(self, "assumptions", assumptions)

    def to_record(self) -> dict[str, object]:
        return {
            "assumptions": list(self.assumptions),
            "interventions": [spec.to_record() for spec in self.interventions],
            "name": self.name,
            "observed_policy": self.observed_policy,
            "schema": "o1fm.intervention-plan/v1",
        }

    @property
    def identity(self) -> str:
        return sha256_json(self.to_record())

    @property
    def canonical_identity(self) -> str:
        return self.identity

    @property
    def edited_covariates(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    index
                    for intervention in self.interventions
                    for index in intervention.edited_covariates
                }
            )
        )


def apply_intervention(
    inputs: ForecastInputs, plan: InterventionPlan
) -> ForecastInputs:
    """Apply ``plan`` to known-future cells and return a fresh input contract."""

    if not isinstance(inputs, ForecastInputs):
        raise TypeError("inputs must be ForecastInputs")
    if not isinstance(plan, InterventionPlan):
        raise TypeError("plan must be InterventionPlan")
    known = np.array(inputs.known_covariates, dtype=np.float32, order="C", copy=True)
    known_mask = np.array(
        inputs.known_covariate_mask,
        dtype=np.bool_,
        order="C",
        copy=True,
    )
    future_ticks = inputs.future_timestamps.astype(np.int64, copy=False)
    row_by_tick = {
        int(tick): inputs.history_length + row for row, tick in enumerate(future_ticks)
    }
    for spec in plan.interventions:
        for local_row, tick in enumerate(spec.timestamps.astype(np.int64, copy=False)):
            try:
                target_row = row_by_tick[int(tick)]
            except KeyError as exc:
                raise ValueError(
                    "intervention timestamp lies outside the exact forecast horizon"
                ) from exc
            for local_column, target_column in enumerate(spec.covariate_indices):
                if target_column >= inputs.known_covariate_dim:
                    raise ValueError(
                        "intervention covariate index is outside the input"
                    )
                if not bool(spec.edit_mask[local_row, local_column]):
                    continue
                previous = float(known[target_row, target_column])
                operand = float(spec.values[local_row, local_column])
                reveal = bool(
                    spec.observed_mask is not None
                    and spec.observed_mask[local_row, local_column]
                )
                if (
                    spec.operation is not InterventionOperation.SET
                    and reveal
                    and not bool(known_mask[target_row, target_column])
                ):
                    raise ValueError(
                        "shift/scale cannot reveal a value whose base is unobserved"
                    )
                if spec.operation is InterventionOperation.SET:
                    updated = operand
                elif spec.operation is InterventionOperation.SHIFT:
                    updated = previous + operand
                else:
                    updated = previous * operand
                lower = spec.lower_bounds[local_column]
                upper = spec.upper_bounds[local_column]
                if not math.isfinite(updated) or not lower <= updated <= upper:
                    raise ValueError("intervention result violates its declared bounds")
                selected = np.float32(updated)
                if not np.isfinite(selected):
                    raise ValueError("intervention result is outside float32")
                known[target_row, target_column] = selected
                if reveal:
                    known_mask[target_row, target_column] = True
    return ForecastInputs(
        series_id=inputs.series_id,
        horizon=inputs.horizon,
        timestamps=inputs.timestamps,
        past_target=inputs.past_target,
        past_target_mask=inputs.past_target_mask,
        past_covariates=inputs.past_covariates,
        past_covariate_mask=inputs.past_covariate_mask,
        known_covariates=known,
        known_covariate_mask=known_mask,
    )


@dataclass(frozen=True, slots=True)
class JointEnvelope:
    """A joint lower/upper trajectory envelope with explicit alignment."""

    series_id: str
    timestamps: object
    lower: object
    upper: object

    def __post_init__(self) -> None:
        series_id = _text(self.series_id, field_name="series_id")
        raw_lower = np.asarray(self.lower)
        if raw_lower.ndim != 2 or raw_lower.shape[0] < 1 or raw_lower.shape[1] < 1:
            raise ValueError("joint envelope must be a non-empty rank-two matrix")
        shape = tuple(int(value) for value in raw_lower.shape)
        lower = _readonly_float32(self.lower, shape=shape, field_name="lower")
        upper = _readonly_float32(self.upper, shape=shape, field_name="upper")
        if bool((lower > upper).any()):
            raise ValueError("joint envelope lower bound exceeds upper bound")
        timestamps = _timestamps(self.timestamps, field_name="envelope timestamps")
        if timestamps.shape != (shape[0],):
            raise ValueError("joint envelope timestamps do not match its horizon")
        object.__setattr__(self, "series_id", series_id)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def validate_for(self, prediction: ForecastPrediction) -> None:
        if (
            self.series_id != prediction.series_id
            or self.lower.shape != prediction.point.shape
            or not np.array_equal(self.timestamps, prediction.timestamps)
        ):
            raise ValueError("joint envelope is not aligned with its prediction")

    @property
    def identity(self) -> str:
        return sha256_json(
            {
                "lower_sha256": _array_digest(self.lower),
                "series_id": self.series_id,
                "shape": list(self.lower.shape),
                "timestamps_ns_sha256": _array_digest(
                    self.timestamps.astype(np.int64, copy=False)
                ),
                "upper_sha256": _array_digest(self.upper),
            }
        )


def _citations(value: object) -> tuple[AtlasReceipt, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("atlas_citations must be a sequence")
    result = tuple(value)
    if not result or not all(isinstance(item, AtlasReceipt) for item in result):
        raise TypeError("each forecast branch requires direct AtlasReceipt citations")
    identities = [
        (item.segment_sha256, item.revision_sequence, item.kind) for item in result
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Atlas citations must be unique")
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.revision_sequence,
                item.kind,
                item.segment_sha256,
            ),
        )
    )


def _citation_records(citations: Sequence[AtlasReceipt]) -> list[dict[str, object]]:
    return [
        {
            "kind": item.kind,
            "record_count": item.record_count,
            "revision_sequence": item.revision_sequence,
            "revision_sha256": item.revision_sha256,
            "segment_sha256": item.segment_sha256,
        }
        for item in citations
    ]


def _edges(value: object) -> tuple[CausalEdge, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("route_edges must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, CausalEdge) for item in result):
        raise TypeError("route_edges must contain CausalEdge instances")
    identities = [
        (edge.source, edge.target, edge.lag, edge.kind, edge.revision_sha256)
        for edge in result
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("route_edges must be unique")
    return tuple(
        sorted(
            result,
            key=lambda edge: (
                edge.source,
                edge.target,
                edge.lag,
                edge.kind,
                edge.revision_sha256,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class BranchExecutionResult:
    """Structural output expected from a branch forecast callable."""

    prediction: ForecastPrediction
    action_identity: str
    model_identity: str
    profile_identity: str
    route_identity: str
    graph_identity: str
    route_w1: float
    atlas_citations: Sequence[AtlasReceipt]
    joint_envelope: JointEnvelope | None = None
    route_edges: Sequence[CausalEdge] = ()
    assumptions: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, ForecastPrediction):
            raise TypeError("prediction must be ForecastPrediction")
        for name in (
            "action_identity",
            "model_identity",
            "profile_identity",
            "route_identity",
            "graph_identity",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), field_name=name))
        route_w1 = _finite(self.route_w1, field_name="route_w1", nonnegative=True)
        citations = _citations(self.atlas_citations)
        edges = _edges(self.route_edges)
        assumptions = _texts(self.assumptions, field_name="assumption")
        if self.joint_envelope is not None:
            if not isinstance(self.joint_envelope, JointEnvelope):
                raise TypeError("joint_envelope must be JointEnvelope or None")
            self.joint_envelope.validate_for(self.prediction)
        object.__setattr__(self, "route_w1", route_w1)
        object.__setattr__(self, "atlas_citations", citations)
        object.__setattr__(self, "route_edges", edges)
        object.__setattr__(self, "assumptions", assumptions)


@dataclass(frozen=True, slots=True)
class ForecastBranch:
    """One immutable forecast trajectory and its complete execution identity."""

    prediction: ForecastPrediction
    identification: IdentificationStatus | str
    action_identity: str
    model_identity: str
    profile_identity: str
    route_identity: str
    graph_identity: str
    route_w1: float
    atlas_citations: Sequence[AtlasReceipt]
    parent_identity: str | None = None
    factual_identity: str | None = None
    intervention_identity: str | None = None
    joint_envelope: JointEnvelope | None = None
    supporting_edges: Sequence[CausalEdge] = ()
    assumptions: Sequence[str] = ()
    branch_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, ForecastPrediction):
            raise TypeError("prediction must be ForecastPrediction")
        try:
            identification = IdentificationStatus(self.identification)
        except ValueError as exc:
            raise ValueError("identification status is invalid") from exc
        for name in (
            "action_identity",
            "model_identity",
            "profile_identity",
            "route_identity",
            "graph_identity",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), field_name=name))
        route_w1 = _finite(self.route_w1, field_name="route_w1", nonnegative=True)
        citations = _citations(self.atlas_citations)
        supporting = _edges(self.supporting_edges)
        assumptions = _texts(self.assumptions, field_name="assumption")
        if self.joint_envelope is not None:
            if not isinstance(self.joint_envelope, JointEnvelope):
                raise TypeError("joint_envelope must be JointEnvelope or None")
            self.joint_envelope.validate_for(self.prediction)
        if identification is IdentificationStatus.FACTUAL:
            if (
                self.parent_identity is not None
                or self.intervention_identity is not None
            ):
                raise ValueError("factual branch cannot have a parent or intervention")
            if self.factual_identity is not None:
                raise ValueError("factual identity is derived from the factual branch")
            factual_record: str = "self"
        else:
            if self.parent_identity is None or self.factual_identity is None:
                raise ValueError(
                    "non-factual branch requires parent and factual identities"
                )
            require_sha256(self.parent_identity, field="parent_identity")
            factual_record = require_sha256(
                self.factual_identity, field="factual_identity"
            )
            if self.intervention_identity is None:
                raise ValueError("non-factual branch requires an intervention identity")
            require_sha256(self.intervention_identity, field="intervention_identity")
        if identification is IdentificationStatus.GRAPH_SUPPORTED_INTERVENTION:
            if not supporting or any(
                edge.kind != "intervention" for edge in supporting
            ):
                raise ValueError(
                    "graph-supported identification requires intervention CausalEdges"
                )
        elif supporting:
            raise ValueError("supporting edges belong only to graph-supported branches")
        if (
            identification is IdentificationStatus.COUNTERFACTUAL_HYPOTHESIS
            and not assumptions
        ):
            raise ValueError("counterfactual hypotheses require explicit assumptions")
        branch_identity = sha256_json(
            {
                "action_identity": self.action_identity,
                "assumptions": list(assumptions),
                "atlas_citations": _citation_records(citations),
                "factual_identity": factual_record,
                "graph_identity": self.graph_identity,
                "identification": identification.value,
                "intervention_identity": self.intervention_identity,
                "joint_envelope_identity": (
                    None
                    if self.joint_envelope is None
                    else self.joint_envelope.identity
                ),
                "model_identity": self.model_identity,
                "parent_identity": self.parent_identity,
                "prediction": _prediction_record(self.prediction),
                "profile_identity": self.profile_identity,
                "route_identity": self.route_identity,
                "route_w1": route_w1.hex(),
                "schema": "o1fm.forecast-branch/v1",
                "supporting_edges": [edge.to_record() for edge in supporting],
            }
        )
        object.__setattr__(self, "identification", identification)
        object.__setattr__(self, "route_w1", route_w1)
        object.__setattr__(self, "atlas_citations", citations)
        object.__setattr__(self, "supporting_edges", supporting)
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "branch_identity", branch_identity)
        if identification is IdentificationStatus.FACTUAL:
            object.__setattr__(self, "factual_identity", branch_identity)

    @property
    def is_factual(self) -> bool:
        return self.identification is IdentificationStatus.FACTUAL


def _prediction_bounds(
    branch: ForecastBranch,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if branch.joint_envelope is not None:
        return branch.joint_envelope.lower, branch.joint_envelope.upper
    if branch.prediction.quantiles is not None:
        return branch.prediction.quantiles[:, :, 0], branch.prediction.quantiles[
            :, :, -1
        ]
    return branch.prediction.point, branch.prediction.point


@dataclass(frozen=True, slots=True)
class TargetTrajectory:
    """One target trajectory retaining every available uncertainty channel."""

    branch_identity: str
    target_index: int
    point: NDArray[np.float32]
    quantiles: NDArray[np.float32] | None
    quantile_levels: NDArray[np.float32] | None
    joint_lower: NDArray[np.float32]
    joint_upper: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class TrajectoryExtremes:
    best: TargetTrajectory
    worst: TargetTrajectory


@dataclass(frozen=True, slots=True)
class BranchSet:
    """An aligned immutable branch family rooted in exactly one factual path."""

    branches: Mapping[str, ForecastBranch] | Sequence[ForecastBranch]

    def __post_init__(self) -> None:
        if isinstance(self.branches, Mapping):
            selected = tuple(self.branches.values())
            if any(
                key != value.branch_identity for key, value in self.branches.items()
            ):
                raise ValueError("branch mapping keys must equal branch identities")
        elif isinstance(self.branches, Sequence) and not isinstance(
            self.branches, (str, bytes)
        ):
            selected = tuple(self.branches)
        else:
            raise TypeError("branches must be a mapping or sequence")
        if not selected or not all(
            isinstance(branch, ForecastBranch) for branch in selected
        ):
            raise TypeError("branches must contain ForecastBranch instances")
        by_id = {branch.branch_identity: branch for branch in selected}
        if len(by_id) != len(selected):
            raise ValueError("branch identities must be unique")
        factual = tuple(branch for branch in selected if branch.is_factual)
        if len(factual) != 1:
            raise ValueError("branch set requires exactly one factual root")
        root = factual[0]
        levels = root.prediction.quantile_levels
        for branch in selected:
            prediction = branch.prediction
            if (
                prediction.series_id != root.prediction.series_id
                or prediction.point.shape != root.prediction.point.shape
                or not np.array_equal(prediction.timestamps, root.prediction.timestamps)
            ):
                raise ValueError("branch predictions are not aligned")
            if (levels is None) != (prediction.quantile_levels is None):
                raise ValueError("branch quantile contracts differ")
            if levels is not None and not np.array_equal(
                levels, prediction.quantile_levels
            ):
                raise ValueError("branch quantile levels differ")
            if branch.factual_identity != root.branch_identity:
                raise ValueError("branch refers to a different factual root")
            if not branch.is_factual and branch.parent_identity not in by_id:
                raise ValueError("branch parent is absent from the branch set")
        object.__setattr__(
            self,
            "branches",
            MappingProxyType(dict(sorted(by_id.items()))),
        )

    @property
    def factual(self) -> ForecastBranch:
        return next(branch for branch in self.branches.values() if branch.is_factual)

    def _branch(self, value: str | ForecastBranch) -> ForecastBranch:
        identity = value.branch_identity if isinstance(value, ForecastBranch) else value
        try:
            return self.branches[identity]
        except KeyError as exc:
            raise KeyError("branch is not a member of this BranchSet") from exc

    def delta(self, branch: str | ForecastBranch) -> NDArray[np.float32]:
        """Return the immutable point trajectory difference from factual."""

        selected = self._branch(branch)
        delta = np.asarray(
            selected.prediction.point - self.factual.prediction.point,
            dtype=np.float32,
            order="C",
        )
        delta.setflags(write=False)
        return delta

    def effect_envelope(self, branch: str | ForecastBranch) -> JointEnvelope:
        """Return the conservative branch-minus-factual uncertainty envelope."""

        selected = self._branch(branch)
        branch_lower, branch_upper = _prediction_bounds(selected)
        factual_lower, factual_upper = _prediction_bounds(self.factual)
        lower = branch_lower - factual_upper
        upper = branch_upper - factual_lower
        return JointEnvelope(
            series_id=self.factual.prediction.series_id,
            timestamps=self.factual.prediction.timestamps,
            lower=lower,
            upper=upper,
        )

    def target_trajectory(
        self,
        branch: str | ForecastBranch,
        target_index: int,
    ) -> TargetTrajectory:
        selected = self._branch(branch)
        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or not 0 <= target_index < selected.prediction.target_dim
        ):
            raise ValueError("target_index is outside the forecast contract")
        lower, upper = _prediction_bounds(selected)

        def vector(value: NDArray[np.float32]) -> NDArray[np.float32]:
            result = np.array(value, dtype=np.float32, order="C", copy=True)
            result.setflags(write=False)
            return result

        quantiles = (
            None
            if selected.prediction.quantiles is None
            else vector(selected.prediction.quantiles[:, target_index, :])
        )
        levels = (
            None
            if selected.prediction.quantile_levels is None
            else vector(selected.prediction.quantile_levels)
        )
        return TargetTrajectory(
            branch_identity=selected.branch_identity,
            target_index=target_index,
            point=vector(selected.prediction.point[:, target_index]),
            quantiles=quantiles,
            quantile_levels=levels,
            joint_lower=vector(lower[:, target_index]),
            joint_upper=vector(upper[:, target_index]),
        )

    def best_worst_target_trajectories(
        self,
        target_index: int,
        *,
        maximize: bool = True,
    ) -> TrajectoryExtremes:
        """Rank complete branches while retaining their uncertainty tensors."""

        if not isinstance(maximize, bool):
            raise TypeError("maximize must be boolean")
        trajectories = tuple(
            self.target_trajectory(branch, target_index)
            for branch in self.branches.values()
        )
        ordered = sorted(
            trajectories,
            key=lambda item: (
                math.fsum(float(value) for value in item.point),
                item.branch_identity,
            ),
        )
        best = ordered[-1] if maximize else ordered[0]
        worst = ordered[0] if maximize else ordered[-1]
        return TrajectoryExtremes(best=best, worst=worst)

    def best_worst(
        self,
        target_index: int,
        *,
        maximize: bool = True,
    ) -> TrajectoryExtremes:
        return self.best_worst_target_trajectories(target_index, maximize=maximize)


@runtime_checkable
class BranchForecastCallable(Protocol):
    def __call__(self, inputs: ForecastInputs) -> BranchExecutionResult: ...


class BranchExecutor:
    """Execute deterministic factual and intervention trajectories."""

    def __init__(
        self,
        forecast: Callable[[ForecastInputs], BranchExecutionResult],
        *,
        known_covariate_names: Sequence[str] | None = None,
        target_names: Sequence[str] | None = None,
    ) -> None:
        if not callable(forecast):
            raise TypeError("forecast must be callable")
        self._forecast = forecast
        self._known_covariate_names = (
            None
            if known_covariate_names is None
            else _texts(known_covariate_names, field_name="known covariate name")
        )
        self._target_names = (
            None
            if target_names is None
            else _texts(target_names, field_name="target name")
        )

    @staticmethod
    def _branch(
        execution: BranchExecutionResult,
        *,
        identification: IdentificationStatus,
        parent_identity: str | None,
        factual_identity: str | None,
        intervention_identity: str | None,
        supporting_edges: Sequence[CausalEdge],
        assumptions: Sequence[str],
    ) -> ForecastBranch:
        return ForecastBranch(
            prediction=execution.prediction,
            identification=identification,
            action_identity=execution.action_identity,
            model_identity=execution.model_identity,
            profile_identity=execution.profile_identity,
            route_identity=execution.route_identity,
            graph_identity=execution.graph_identity,
            route_w1=execution.route_w1,
            atlas_citations=execution.atlas_citations,
            parent_identity=parent_identity,
            factual_identity=factual_identity,
            intervention_identity=intervention_identity,
            joint_envelope=execution.joint_envelope,
            supporting_edges=supporting_edges,
            assumptions=assumptions,
        )

    def _names(self, inputs: ForecastInputs) -> tuple[tuple[str, ...], tuple[str, ...]]:
        known = self._known_covariate_names or tuple(
            f"known:{index}" for index in range(inputs.known_covariate_dim)
        )
        targets = self._target_names or tuple(
            f"target:{index}" for index in range(inputs.target_dim)
        )
        if len(known) != inputs.known_covariate_dim:
            raise ValueError("known covariate names do not match the input contract")
        if len(targets) != inputs.target_dim:
            raise ValueError("target names do not match the input contract")
        if len(set(known)) != len(known) or len(set(targets)) != len(targets):
            raise ValueError("branch variable names must be unique")
        return known, targets

    def execute(
        self,
        inputs: ForecastInputs,
        plans: Sequence[InterventionPlan] = (),
    ) -> BranchSet:
        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        if isinstance(plans, (str, bytes)) or not isinstance(plans, Sequence):
            raise TypeError("plans must be a sequence")
        selected_plans = tuple(plans)
        if not all(isinstance(plan, InterventionPlan) for plan in selected_plans):
            raise TypeError("plans must contain InterventionPlan instances")
        plan_ids = [plan.identity for plan in selected_plans]
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("intervention plan identities must be unique")
        selected_plans = tuple(sorted(selected_plans, key=lambda item: item.identity))
        known_names, target_names = self._names(inputs)

        factual_execution = self._forecast(inputs)
        if not isinstance(factual_execution, BranchExecutionResult):
            raise TypeError("forecast callable must return BranchExecutionResult")
        factual_execution.prediction.validate_for(inputs)
        factual = self._branch(
            factual_execution,
            identification=IdentificationStatus.FACTUAL,
            parent_identity=None,
            factual_identity=None,
            intervention_identity=None,
            supporting_edges=(),
            assumptions=factual_execution.assumptions,
        )
        branches = [factual]
        for plan in selected_plans:
            branch_inputs = apply_intervention(inputs, plan)
            execution = self._forecast(branch_inputs)
            if not isinstance(execution, BranchExecutionResult):
                raise TypeError("forecast callable must return BranchExecutionResult")
            execution.prediction.validate_for(branch_inputs)
            sources = {known_names[index] for index in plan.edited_covariates}
            matching = tuple(
                edge
                for edge in execution.route_edges
                if edge.kind == "intervention"
                and edge.lag == 0
                and edge.source in sources
                and edge.target in target_names
            )
            required_pairs = {
                (source, target) for source in sources for target in target_names
            }
            covered_pairs = {(edge.source, edge.target) for edge in matching}
            if plan.observed_policy:
                identification = IdentificationStatus.OBSERVED_POLICY_INTERVENTION
                supporting: tuple[CausalEdge, ...] = ()
                extra_assumptions: tuple[str, ...] = ()
            elif required_pairs and covered_pairs == required_pairs:
                identification = IdentificationStatus.GRAPH_SUPPORTED_INTERVENTION
                supporting = matching
                extra_assumptions = ()
            else:
                identification = IdentificationStatus.COUNTERFACTUAL_HYPOTHESIS
                supporting = ()
                missing = (
                    ",".join(
                        f"{source}->{target}"
                        for source, target in sorted(required_pairs - covered_pairs)
                    )
                    or "none"
                )
                extra_assumptions = (
                    f"No complete intervention-edge support for pairs: {missing}",
                )
            branches.append(
                self._branch(
                    execution,
                    identification=identification,
                    parent_identity=factual.branch_identity,
                    factual_identity=factual.branch_identity,
                    intervention_identity=plan.identity,
                    supporting_edges=supporting,
                    assumptions=(
                        *plan.assumptions,
                        *execution.assumptions,
                        *extra_assumptions,
                    ),
                )
            )
        return BranchSet(branches)


__all__ = [
    "BranchExecutionResult",
    "BranchExecutor",
    "BranchForecastCallable",
    "BranchSet",
    "ForecastBranch",
    "IdentificationStatus",
    "InterventionOperation",
    "InterventionPlan",
    "InterventionSpec",
    "JointEnvelope",
    "TargetTrajectory",
    "TrajectoryExtremes",
    "apply_intervention",
]
