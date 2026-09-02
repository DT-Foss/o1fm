"""Compiled intervention response surfaces over factual teacher trajectories."""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from .contracts import ForecastPrediction, ForecastTarget
from .crystals import ForecastCrystalBank, ForecastCrystalKey
from .identity import canonical_json_bytes, require_sha256, sha256_json

SCHEMA = "o1fm.intervention-response-bank/v1"
KEY_SCHEMA = "o1fm.intervention-response-key/v1"
OBSERVATION_SCHEMA = "o1fm.intervention-response-observation/v1"
MAX_STATE_BYTES = 128 * 1024 * 1024
MAX_TEXT_BYTES = 4096
MAX_FEATURES = 4096


class InterventionResponseError(RuntimeError):
    """An intervention response cannot be compiled or executed."""


class InterventionResponseConflictError(InterventionResponseError):
    """One observation identity was reused with different response evidence."""


class InterventionResponseIntegrityError(InterventionResponseError):
    """Persisted response state is malformed or internally inconsistent."""


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > MAX_TEXT_BYTES
    ):
        raise ValueError(f"{field} must be canonical bounded text")
    return value


def _positive_int(value: object, *, field: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _finite(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{field} is outside its finite bound")
    return 0.0 if result == 0.0 else result


def _features(value: object) -> NDArray[np.float64]:
    try:
        result = np.array(value, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("features must be numeric") from exc
    if (
        result.ndim != 1
        or not 1 <= result.size <= MAX_FEATURES
        or not bool(np.isfinite(result).all())
    ):
        raise ValueError("features must be a bounded finite vector")
    result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _levels(value: Sequence[object]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("quantile_levels must be a sequence")
    levels = tuple(_finite(item, field="quantile level") for item in value)
    if (
        not levels
        or any(not 0.0 < item < 1.0 for item in levels)
        or any(right <= left for left, right in zip(levels, levels[1:], strict=False))
    ):
        raise ValueError("quantile levels must be strictly ordered inside (0, 1)")
    return levels


def _atomic_write(path: Path, document: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(document)
    if len(payload) > MAX_STATE_BYTES:
        raise InterventionResponseIntegrityError("response state exceeds its bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class InterventionResponseKey:
    """One exact state/route/regime response surface."""

    model_identity: str
    feature_schema: str
    intervention_schema: str
    target_schema: str
    graph_revision: str
    regime: str
    horizon: int
    target_count: int

    def __post_init__(self) -> None:
        for field in (
            "model_identity",
            "feature_schema",
            "intervention_schema",
            "target_schema",
            "graph_revision",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        object.__setattr__(self, "regime", _text(self.regime, field="regime"))
        object.__setattr__(
            self,
            "horizon",
            _positive_int(self.horizon, field="horizon"),
        )
        object.__setattr__(
            self,
            "target_count",
            _positive_int(self.target_count, field="target_count"),
        )

    @property
    def output_dim(self) -> int:
        return self.horizon * self.target_count

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "feature_schema": self.feature_schema,
            "graph_revision": self.graph_revision,
            "horizon": self.horizon,
            "intervention_schema": self.intervention_schema,
            "model_identity": self.model_identity,
            "regime": self.regime,
            "schema": KEY_SCHEMA,
            "target_count": self.target_count,
            "target_schema": self.target_schema,
        }

    @classmethod
    def from_record(cls, value: object) -> InterventionResponseKey:
        if not isinstance(value, Mapping) or set(value) != {
            "feature_schema",
            "graph_revision",
            "horizon",
            "intervention_schema",
            "model_identity",
            "regime",
            "schema",
            "target_count",
            "target_schema",
        }:
            raise InterventionResponseIntegrityError("response key record is invalid")
        if value["schema"] != KEY_SCHEMA:
            raise InterventionResponseIntegrityError(
                "response key schema is unsupported"
            )
        return cls(
            model_identity=value["model_identity"],  # type: ignore[arg-type]
            feature_schema=value["feature_schema"],  # type: ignore[arg-type]
            intervention_schema=value["intervention_schema"],  # type: ignore[arg-type]
            target_schema=value["target_schema"],  # type: ignore[arg-type]
            graph_revision=value["graph_revision"],  # type: ignore[arg-type]
            regime=value["regime"],  # type: ignore[arg-type]
            horizon=value["horizon"],  # type: ignore[arg-type]
            target_count=value["target_count"],  # type: ignore[arg-type]
        )

    def crystal_key(self) -> ForecastCrystalKey:
        return ForecastCrystalKey(
            model_identity=self.model_identity,
            feature_schema=sha256_json(
                {
                    "feature_schema": self.feature_schema,
                    "intervention_schema": self.intervention_schema,
                    "schema": "o1fm.intervention-response-inner-schema/v1",
                    "target_schema": self.target_schema,
                }
            ),
            graph_revision=self.graph_revision,
            regime=self.regime,
            horizon=self.horizon,
            target_count=self.target_count,
        )


@dataclass(frozen=True, slots=True)
class InterventionResponseStats:
    key: InterventionResponseKey
    feature_dim: int
    support: int
    sequential_support: int
    observed_values: int
    mean_effect_error: float | None
    effect_direction_cosine: float | None
    empirical_coverage: float | None
    ready: bool


@dataclass(frozen=True, slots=True)
class InterventionResponsePrediction:
    """A compiled effect and the corrected intervention trajectory."""

    key: InterventionResponseKey
    effect_point: NDArray[np.float32]
    effect_quantiles: Mapping[float, NDArray[np.float32]]
    corrected: ForecastPrediction
    stats: InterventionResponseStats

    def __post_init__(self) -> None:
        shape = (self.key.horizon, self.key.target_count)
        if self.effect_point.shape != shape or self.effect_point.dtype != np.float32:
            raise ValueError("effect point differs from the response key")
        if not bool(np.isfinite(self.effect_point).all()):
            raise ValueError("effect point must be finite")
        quantiles = dict(self.effect_quantiles)
        if not quantiles or any(
            value.shape != shape
            or value.dtype != np.float32
            or not bool(np.isfinite(value).all())
            for value in quantiles.values()
        ):
            raise ValueError("effect quantiles differ from the response key")
        if not isinstance(self.corrected, ForecastPrediction):
            raise TypeError("corrected must be ForecastPrediction")
        object.__setattr__(
            self,
            "effect_quantiles",
            MappingProxyType(dict(sorted(quantiles.items()))),
        )


@dataclass(slots=True)
class _MetricState:
    key: InterventionResponseKey
    feature_dim: int
    sequential_support: int = 0
    observed_values: int = 0
    absolute_error_sum: float = 0.0
    dot_sum: float = 0.0
    predicted_square_sum: float = 0.0
    actual_square_sum: float = 0.0
    covered_values: int = 0

    def to_record(self) -> dict[str, object]:
        return {
            "absolute_error_sum": self.absolute_error_sum,
            "actual_square_sum": self.actual_square_sum,
            "covered_values": self.covered_values,
            "dot_sum": self.dot_sum,
            "feature_dim": self.feature_dim,
            "key": self.key.to_record(),
            "observed_values": self.observed_values,
            "predicted_square_sum": self.predicted_square_sum,
            "sequential_support": self.sequential_support,
        }

    @classmethod
    def from_record(cls, value: object) -> _MetricState:
        if not isinstance(value, Mapping):
            raise InterventionResponseIntegrityError("metric entry is invalid")
        key = InterventionResponseKey.from_record(value.get("key"))
        state = cls(
            key=key,
            feature_dim=_positive_int(value.get("feature_dim"), field="feature_dim"),
            sequential_support=_positive_int(
                value.get("sequential_support"),
                field="sequential_support",
                allow_zero=True,
            ),
            observed_values=_positive_int(
                value.get("observed_values"), field="observed_values", allow_zero=True
            ),
            absolute_error_sum=_finite(
                value.get("absolute_error_sum"),
                field="absolute_error_sum",
                minimum=0.0,
            ),
            dot_sum=_finite(value.get("dot_sum"), field="dot_sum"),
            predicted_square_sum=_finite(
                value.get("predicted_square_sum"),
                field="predicted_square_sum",
                minimum=0.0,
            ),
            actual_square_sum=_finite(
                value.get("actual_square_sum"),
                field="actual_square_sum",
                minimum=0.0,
            ),
            covered_values=_positive_int(
                value.get("covered_values"),
                field="covered_values",
                allow_zero=True,
            ),
        )
        if state.covered_values > state.observed_values:
            raise InterventionResponseIntegrityError("coverage counts are inconsistent")
        return state


class InterventionResponseCrystal:
    """Persistent ridge compiler for intervention effects over a factual prior."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        ridge: float = 1.0,
        min_support: int = 8,
        min_sequential_support: int = 4,
        max_mean_effect_error: float | None = None,
        min_direction_cosine: float = 0.0,
        min_coverage: float | None = None,
        quantile_levels: Sequence[object] = (0.1, 0.5, 0.9),
        max_residual_history: int = 2048,
    ) -> None:
        path = Path(root).expanduser().absolute()
        if path.suffix == ".json":
            self.state_path = path
            effect_path = path.with_name(f"{path.stem}.effects.json")
        else:
            self.state_path = path / "intervention_response.json"
            effect_path = path / "effect_crystals.json"
        self.quantile_levels = _levels(quantile_levels)
        self.min_support = _positive_int(min_support, field="min_support")
        self.min_sequential_support = _positive_int(
            min_sequential_support,
            field="min_sequential_support",
            allow_zero=True,
        )
        self.max_mean_effect_error = (
            None
            if max_mean_effect_error is None
            else _finite(
                max_mean_effect_error,
                field="max_mean_effect_error",
                minimum=0.0,
            )
        )
        self.min_direction_cosine = _finite(
            min_direction_cosine,
            field="min_direction_cosine",
            minimum=-1.0,
            maximum=1.0,
        )
        self.min_coverage = (
            None
            if min_coverage is None
            else _finite(min_coverage, field="min_coverage", minimum=0.0, maximum=1.0)
        )
        self.effects = ForecastCrystalBank(
            effect_path,
            ridge=ridge,
            min_support=self.min_support,
            # Promotion authority belongs to the response wrapper.  The inner
            # bank becomes queryable at fit support so later observations can
            # accumulate strictly pre-update direction/coverage evidence.
            min_sequential_support=0,
            max_mean_absolute_error=None,
            min_coverage=None,
            quantile_levels=self.quantile_levels,
            max_residual_history=max_residual_history,
        )
        self._states: dict[str, _MetricState] = {}
        self._observations: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()
        if self.state_path.exists():
            self._load()

    def _load(self) -> None:
        payload = self.state_path.read_bytes()
        if not payload or len(payload) > MAX_STATE_BYTES:
            raise InterventionResponseIntegrityError("response state size is invalid")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InterventionResponseIntegrityError(
                "response state is not JSON"
            ) from exc
        if (
            not isinstance(document, Mapping)
            or document.get("schema") != SCHEMA
            or not isinstance(document.get("entries"), list)
            or not isinstance(document.get("observations"), Mapping)
        ):
            raise InterventionResponseIntegrityError(
                "response state envelope is invalid"
            )
        if canonical_json_bytes(document) != payload:
            raise InterventionResponseIntegrityError("response state is not canonical")
        states: dict[str, _MetricState] = {}
        for raw in document["entries"]:
            state = _MetricState.from_record(raw)
            identity = state.key.identity_sha256
            if identity in states:
                raise InterventionResponseIntegrityError("response key is duplicated")
            states[identity] = state
        observations: dict[str, dict[str, object]] = {}
        for raw_id, raw in document["observations"].items():
            selected_id = _text(raw_id, field="observation_id")
            if not isinstance(raw, Mapping) or set(raw) != {
                "committed",
                "content_sha256",
                "key_sha256",
                "metric",
            }:
                raise InterventionResponseIntegrityError(
                    "observation record is invalid"
                )
            require_sha256(raw["content_sha256"], field="content_sha256")
            require_sha256(raw["key_sha256"], field="key_sha256")
            if not isinstance(raw["committed"], bool):
                raise InterventionResponseIntegrityError(
                    "observation status is invalid"
                )
            observations[selected_id] = dict(raw)
        self._states = states
        self._observations = observations

    def _persist(self) -> None:
        _atomic_write(
            self.state_path,
            {
                "entries": [
                    self._states[key].to_record() for key in sorted(self._states)
                ],
                "observations": {
                    key: self._observations[key] for key in sorted(self._observations)
                },
                "schema": SCHEMA,
            },
        )

    def _stats(self, key: InterventionResponseKey) -> InterventionResponseStats:
        effect_stats = self.effects.describe(key.crystal_key())
        support = 0 if effect_stats is None else effect_stats.support
        state = self._states.get(key.identity_sha256)
        feature_dim = 0 if state is None else state.feature_dim
        sequential = 0 if state is None else state.sequential_support
        observed = 0 if state is None else state.observed_values
        mae = (
            None
            if state is None or state.observed_values == 0
            else state.absolute_error_sum / state.observed_values
        )
        if state is None or state.observed_values == 0:
            cosine = None
            coverage = None
        else:
            denominator = math.sqrt(
                state.predicted_square_sum * state.actual_square_sum
            )
            if denominator > 0.0:
                cosine = max(-1.0, min(1.0, state.dot_sum / denominator))
            elif state.predicted_square_sum == 0.0 and state.actual_square_sum == 0.0:
                cosine = 1.0
            else:
                cosine = 0.0
            coverage = state.covered_values / state.observed_values
        ready = (
            support >= self.min_support
            and sequential >= self.min_sequential_support
            and mae is not None
            and (
                self.max_mean_effect_error is None or mae <= self.max_mean_effect_error
            )
            and cosine is not None
            and cosine >= self.min_direction_cosine
            and (
                self.min_coverage is None
                or (coverage is not None and coverage >= self.min_coverage)
            )
        )
        return InterventionResponseStats(
            key=key,
            feature_dim=feature_dim,
            support=support,
            sequential_support=sequential,
            observed_values=observed,
            mean_effect_error=mae,
            effect_direction_cosine=cosine,
            empirical_coverage=coverage,
            ready=ready,
        )

    def describe(self, key: InterventionResponseKey) -> InterventionResponseStats:
        if not isinstance(key, InterventionResponseKey):
            raise TypeError("key must be InterventionResponseKey")
        with self._lock:
            return self._stats(key)

    def predict(
        self,
        key: InterventionResponseKey,
        features: object,
        factual: ForecastPrediction,
    ) -> InterventionResponsePrediction | None:
        if not isinstance(key, InterventionResponseKey):
            raise TypeError("key must be InterventionResponseKey")
        if not isinstance(factual, ForecastPrediction):
            raise TypeError("factual must be ForecastPrediction")
        if factual.quantile_levels is None:
            raise ValueError("factual prediction requires quantiles")
        factual_levels = tuple(float(value) for value in factual.quantile_levels)
        if (
            factual.horizon != key.horizon
            or factual.target_dim != key.target_count
            or len(factual_levels) != len(self.quantile_levels)
            or any(
                not math.isclose(left, right, abs_tol=1e-7)
                for left, right in zip(
                    factual_levels,
                    self.quantile_levels,
                    strict=True,
                )
            )
        ):
            raise ValueError("factual prediction differs from the response contract")
        vector = _features(features)
        with self._lock:
            effect = self.effects.predict(key.crystal_key(), vector)
            if effect is None:
                return None
            levels = tuple(sorted(effect.quantiles))
            effect_quantiles = MappingProxyType(
                {level: effect.quantiles[level] for level in levels}
            )
            factual_quantiles = factual.quantiles
            assert factual_quantiles is not None
            corrected_quantiles = np.stack(
                [
                    factual_quantiles[..., index] + effect_quantiles[level]
                    for index, level in enumerate(levels)
                ],
                axis=2,
            )
            corrected = ForecastPrediction(
                series_id=factual.series_id,
                timestamps=factual.timestamps,
                point=factual.point + effect.point,
                quantiles=corrected_quantiles,
                quantile_levels=levels,
            )
            return InterventionResponsePrediction(
                key=key,
                effect_point=effect.point,
                effect_quantiles=effect_quantiles,
                corrected=corrected,
                stats=self._stats(key),
            )

    @staticmethod
    def _aligned_targets(
        key: InterventionResponseKey,
        factual: ForecastTarget,
        intervention: ForecastTarget,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        if not isinstance(factual, ForecastTarget) or not isinstance(
            intervention, ForecastTarget
        ):
            raise TypeError("factual and intervention must be ForecastTarget")
        if (
            factual.series_id != intervention.series_id
            or factual.horizon != key.horizon
            or factual.target_dim != key.target_count
            or intervention.horizon != key.horizon
            or intervention.target_dim != key.target_count
            or not np.array_equal(factual.timestamps, intervention.timestamps)
        ):
            raise ValueError("paired targets differ from the response contract")
        mask = factual.mask & intervention.mask
        if not bool(mask.any()):
            raise ValueError("paired response observes no common target value")
        effect = np.asarray(intervention.values, dtype=np.float64) - np.asarray(
            factual.values, dtype=np.float64
        )
        effect[~mask] = 0.0
        return effect, mask

    def _metric(
        self,
        prediction: InterventionResponsePrediction | None,
        effect: NDArray[np.float64],
        mask: NDArray[np.bool_],
    ) -> dict[str, object] | None:
        if prediction is None:
            return None
        actual = effect[mask]
        predicted = np.asarray(prediction.effect_point, dtype=np.float64)[mask]
        levels = tuple(sorted(prediction.effect_quantiles))
        lower = np.asarray(prediction.effect_quantiles[levels[0]], dtype=np.float64)[
            mask
        ]
        upper = np.asarray(prediction.effect_quantiles[levels[-1]], dtype=np.float64)[
            mask
        ]
        return {
            "absolute_error_sum": float(np.abs(actual - predicted).sum()),
            "actual_square_sum": float(actual @ actual),
            "covered_values": int(((actual >= lower) & (actual <= upper)).sum()),
            "dot_sum": float(actual @ predicted),
            "observed_values": int(actual.size),
            "predicted_square_sum": float(predicted @ predicted),
        }

    @staticmethod
    def _apply_metric(state: _MetricState, metric: Mapping[str, object] | None) -> None:
        if metric is None:
            return
        state.sequential_support += 1
        state.observed_values += int(metric["observed_values"])
        state.absolute_error_sum += float(metric["absolute_error_sum"])
        state.dot_sum += float(metric["dot_sum"])
        state.predicted_square_sum += float(metric["predicted_square_sum"])
        state.actual_square_sum += float(metric["actual_square_sum"])
        state.covered_values += int(metric["covered_values"])

    def observe(
        self,
        key: InterventionResponseKey,
        features: object,
        factual_prediction: ForecastPrediction,
        factual_target: ForecastTarget,
        intervention_target: ForecastTarget,
        *,
        observation_id: str,
    ) -> InterventionResponseStats:
        """Settle one paired effect; future targets never enter ``predict``."""

        if not isinstance(key, InterventionResponseKey):
            raise TypeError("key must be InterventionResponseKey")
        selected_id = _text(observation_id, field="observation_id")
        vector = _features(features)
        effect, mask = self._aligned_targets(key, factual_target, intervention_target)
        with self._lock:
            prior = self._observations.get(selected_id)
            content_sha256 = sha256_json(
                {
                    "effect": effect.tolist(),
                    "features": vector.tolist(),
                    "key": key.to_record(),
                    "mask": mask.tolist(),
                    "schema": OBSERVATION_SCHEMA,
                }
            )
            if prior is not None:
                if prior["content_sha256"] != content_sha256:
                    raise InterventionResponseConflictError(
                        "observation_id is bound to different response evidence"
                    )
                if bool(prior["committed"]):
                    return self._stats(key)
                metric = prior["metric"]
            else:
                prediction = self.predict(key, vector, factual_prediction)
                metric = self._metric(prediction, effect, mask)
                self._observations[selected_id] = {
                    "committed": False,
                    "content_sha256": content_sha256,
                    "key_sha256": key.identity_sha256,
                    "metric": metric,
                }
                self._persist()

            self.effects.observe(
                key.crystal_key(),
                vector,
                effect,
                future_target_mask=mask,
                observation_id=f"response:{selected_id}",
            )
            state = self._states.get(key.identity_sha256)
            if state is None:
                state = _MetricState(key=key, feature_dim=int(vector.size))
                self._states[key.identity_sha256] = state
            elif state.feature_dim != vector.size:
                raise InterventionResponseConflictError(
                    "features differ from the response feature schema"
                )
            if metric is not None and not isinstance(metric, Mapping):
                raise InterventionResponseIntegrityError("pending metric is invalid")
            self._apply_metric(state, metric)
            self._observations[selected_id]["committed"] = True
            self._persist()
            return self._stats(key)


__all__ = [
    "InterventionResponseConflictError",
    "InterventionResponseCrystal",
    "InterventionResponseError",
    "InterventionResponseIntegrityError",
    "InterventionResponseKey",
    "InterventionResponsePrediction",
    "InterventionResponseStats",
]
