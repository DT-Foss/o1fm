"""Forecast-native stored compute with deterministic ridge statistics."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

FloatVector = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]

_SCHEMA = "o1fm.forecast-crystal-bank/v1"
_MAX_TEXT_BYTES = 1024
_MAX_FEATURES = 4096
_MAX_OUTPUTS = 16384


def _canonical_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_TEXT_BYTES
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field} must be a positive bounded integer")
    return value


def _nonnegative_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{field} must be a non-negative bounded integer")
    return value


def _finite_float(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _readonly_float32(
    value: NDArray[np.float64], *, shape: tuple[int, ...]
) -> NDArray[np.float32]:
    array = np.array(value, dtype=np.float32, order="C", copy=True)
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError("forecast array could not be canonicalized")
    array.setflags(write=False)
    return array


def _feature_vector(value: object, *, expected_size: int | None = None) -> FloatVector:
    try:
        vector = np.array(value, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("features must be numeric") from exc
    if vector.ndim != 1 or vector.size < 1 or vector.size > _MAX_FEATURES:
        raise ValueError("features must be a bounded rank-one vector")
    if expected_size is not None and vector.size != expected_size:
        raise ValueError("features do not match the crystal feature schema")
    if not bool(np.isfinite(vector).all()):
        raise ValueError("features must be finite")
    return vector


def _future_matrix(
    value: object,
    *,
    horizon: int,
    target_count: int,
    mask: object | None,
) -> FloatMatrix:
    raw = np.asarray(value)
    try:
        matrix = np.array(value, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("future targets must be numeric") from exc
    if matrix.shape != (horizon, target_count):
        raise ValueError("future targets must match [horizon, target_count]")
    observed = _future_mask(mask, horizon=horizon, target_count=target_count)
    if bool((observed & ~np.isfinite(matrix)).any()):
        raise ValueError("future targets must be finite where observed")
    matrix[~observed] = 0.0
    if not bool(np.isfinite(matrix).all()) or raw.shape != matrix.shape:
        raise ValueError("future targets could not be canonicalized")
    return matrix


def _future_mask(
    value: object | None,
    *,
    horizon: int,
    target_count: int,
) -> NDArray[np.bool_]:
    shape = (horizon, target_count)
    if value is None:
        mask = np.ones(shape, dtype=np.bool_, order="C")
    else:
        raw = np.asarray(value)
        mask = np.array(value, dtype=np.bool_, order="C", copy=True)
        if raw.dtype != np.dtype(np.bool_) or mask.shape != shape:
            raise ValueError("future_target_mask must match [horizon, target_count]")
    mask.setflags(write=False)
    return mask


def _quantile_levels(value: Sequence[object]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("quantile_levels must be a sequence")
    levels = tuple(
        sorted({_finite_float(level, field="quantile level") for level in value})
    )
    if not levels:
        raise ValueError("quantile_levels must contain at least one level")
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("quantile_levels must lie strictly inside (0, 1)")
    return levels


def _atomic_json_dump(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _json_load(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError("forecast crystal state is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("forecast crystal state must be a JSON object")
    return document


@dataclass(frozen=True, slots=True)
class ForecastCrystalKey:
    """One immutable executable regime for one exact forecast profile."""

    model_identity: str
    feature_schema: str
    graph_revision: str
    regime: str
    horizon: int
    target_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_identity",
            _canonical_text(self.model_identity, field="model_identity"),
        )
        object.__setattr__(
            self,
            "feature_schema",
            _canonical_text(self.feature_schema, field="feature_schema"),
        )
        object.__setattr__(
            self,
            "graph_revision",
            _canonical_text(self.graph_revision, field="graph_revision"),
        )
        object.__setattr__(
            self,
            "regime",
            _canonical_text(self.regime, field="regime"),
        )
        object.__setattr__(
            self,
            "horizon",
            _positive_int(self.horizon, field="horizon", maximum=4096),
        )
        object.__setattr__(
            self,
            "target_count",
            _positive_int(self.target_count, field="target_count", maximum=4096),
        )

    @property
    def output_dim(self) -> int:
        return self.horizon * self.target_count

    def to_document(self) -> dict[str, object]:
        return {
            "feature_schema": self.feature_schema,
            "graph_revision": self.graph_revision,
            "horizon": self.horizon,
            "model_identity": self.model_identity,
            "regime": self.regime,
            "target_count": self.target_count,
        }


@dataclass(frozen=True, slots=True)
class ForecastCrystalStats:
    """Promotion and verification state for one stored crystal."""

    key: ForecastCrystalKey
    feature_dim: int
    support: int
    sequential_support: int
    residual_support: int
    mean_absolute_error: float | None
    empirical_coverage: float | None
    ready: bool

    def __post_init__(self) -> None:
        if not isinstance(self.key, ForecastCrystalKey):
            raise TypeError("key must be ForecastCrystalKey")
        object.__setattr__(
            self,
            "feature_dim",
            _positive_int(self.feature_dim, field="feature_dim", maximum=_MAX_FEATURES),
        )
        object.__setattr__(
            self,
            "support",
            _nonnegative_int(self.support, field="support", maximum=10**9),
        )
        object.__setattr__(
            self,
            "sequential_support",
            _nonnegative_int(
                self.sequential_support,
                field="sequential_support",
                maximum=10**9,
            ),
        )
        object.__setattr__(
            self,
            "residual_support",
            _nonnegative_int(
                self.residual_support,
                field="residual_support",
                maximum=10**9,
            ),
        )
        if self.mean_absolute_error is not None:
            object.__setattr__(
                self,
                "mean_absolute_error",
                _finite_float(
                    self.mean_absolute_error,
                    field="mean_absolute_error",
                    minimum=0.0,
                ),
            )
        if self.empirical_coverage is not None:
            coverage = _finite_float(
                self.empirical_coverage,
                field="empirical_coverage",
                minimum=0.0,
            )
            if coverage > 1.0:
                raise ValueError("empirical_coverage must not exceed 1")
            object.__setattr__(self, "empirical_coverage", coverage)
        object.__setattr__(self, "ready", bool(self.ready))


@dataclass(frozen=True, slots=True)
class ForecastCrystalPrediction:
    """One deterministic crystal forecast with ordered quantiles."""

    key: ForecastCrystalKey
    point: NDArray[np.float32]
    quantiles: Mapping[float, NDArray[np.float32]]
    support: int
    sequential_support: int
    mean_absolute_error: float | None
    empirical_coverage: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, ForecastCrystalKey):
            raise TypeError("key must be ForecastCrystalKey")
        expected_shape = (self.key.horizon, self.key.target_count)
        if not isinstance(self.point, np.ndarray) or self.point.shape != expected_shape:
            raise ValueError("point forecast shape differs from key")
        if self.point.dtype != np.float32 or not bool(np.isfinite(self.point).all()):
            raise ValueError("point forecast must be finite float32")
        ordered: dict[float, NDArray[np.float32]] = {}
        previous: NDArray[np.float32] | None = None
        for raw_level, raw_values in self.quantiles.items():
            level = _finite_float(raw_level, field="quantile level")
            if not 0.0 < level < 1.0:
                raise ValueError("quantile levels must lie strictly inside (0, 1)")
            if level in ordered:
                raise ValueError("quantile levels must be unique")
            if (
                not isinstance(raw_values, np.ndarray)
                or raw_values.shape != expected_shape
                or raw_values.dtype != np.float32
                or not bool(np.isfinite(raw_values).all())
            ):
                raise ValueError("quantile forecast must be finite float32")
            if previous is not None and bool((raw_values < previous).any()):
                raise ValueError("quantile forecasts must be ordered")
            ordered[level] = raw_values
            previous = raw_values
        if not ordered:
            raise ValueError("quantiles must not be empty")
        object.__setattr__(self, "quantiles", MappingProxyType(ordered))
        object.__setattr__(
            self,
            "support",
            _nonnegative_int(self.support, field="support", maximum=10**9),
        )
        object.__setattr__(
            self,
            "sequential_support",
            _nonnegative_int(
                self.sequential_support,
                field="sequential_support",
                maximum=10**9,
            ),
        )
        if self.mean_absolute_error is not None:
            object.__setattr__(
                self,
                "mean_absolute_error",
                _finite_float(
                    self.mean_absolute_error,
                    field="mean_absolute_error",
                    minimum=0.0,
                ),
            )
        if self.empirical_coverage is not None:
            coverage = _finite_float(
                self.empirical_coverage,
                field="empirical_coverage",
                minimum=0.0,
            )
            if coverage > 1.0:
                raise ValueError("empirical_coverage must not exceed 1")
            object.__setattr__(self, "empirical_coverage", coverage)


@dataclass(slots=True)
class _CrystalState:
    key: ForecastCrystalKey
    feature_dim: int
    support: int
    feature_sum: FloatVector
    target_sum: FloatVector
    gram: FloatMatrix
    cross: FloatMatrix
    residual_rows: list[list[float]]
    sequential_support: int
    absolute_error_sum: float
    covered_values: int
    observed_values: int

    @property
    def residual_support(self) -> int:
        return len(self.residual_rows)

    def mean_absolute_error(self) -> float | None:
        if self.observed_values == 0:
            return None
        return self.absolute_error_sum / self.observed_values

    def empirical_coverage(self) -> float | None:
        if self.observed_values == 0:
            return None
        return self.covered_values / self.observed_values

    def to_document(self) -> dict[str, object]:
        return {
            "absolute_error_sum": self.absolute_error_sum,
            "covered_values": self.covered_values,
            "cross": self.cross.tolist(),
            "feature_dim": self.feature_dim,
            "feature_sum": self.feature_sum.tolist(),
            "gram": self.gram.tolist(),
            "key": self.key.to_document(),
            "observed_values": self.observed_values,
            "residual_rows": self.residual_rows,
            "sequential_support": self.sequential_support,
            "support": self.support,
            "target_sum": self.target_sum.tolist(),
        }

    @classmethod
    def from_document(cls, value: Mapping[str, object]) -> _CrystalState:
        if not isinstance(value, Mapping):
            raise ValueError("forecast crystal entry must be a mapping")
        key = ForecastCrystalKey(**dict(value["key"]))  # type: ignore[arg-type]
        feature_dim = _positive_int(
            value.get("feature_dim"),
            field="feature_dim",
            maximum=_MAX_FEATURES,
        )
        support = _nonnegative_int(value.get("support"), field="support", maximum=10**9)
        feature_sum = _feature_vector(
            value.get("feature_sum"), expected_size=feature_dim
        )
        target_sum = _feature_vector(
            value.get("target_sum"), expected_size=key.output_dim
        )
        gram = np.array(value.get("gram"), dtype=np.float64, order="C", copy=True)
        cross = np.array(value.get("cross"), dtype=np.float64, order="C", copy=True)
        if gram.shape != (feature_dim, feature_dim) or not bool(
            np.isfinite(gram).all()
        ):
            raise ValueError("stored crystal gram is invalid")
        if cross.shape != (feature_dim, key.output_dim) or not bool(
            np.isfinite(cross).all()
        ):
            raise ValueError("stored crystal cross term is invalid")
        residual_rows_raw = value.get("residual_rows")
        if not isinstance(residual_rows_raw, list):
            raise ValueError("stored crystal residual_rows are invalid")
        residual_rows: list[list[float]] = []
        for row in residual_rows_raw:
            vector = _feature_vector(row, expected_size=key.output_dim)
            residual_rows.append([float(item) for item in vector.tolist()])
        sequential_support = _nonnegative_int(
            value.get("sequential_support"),
            field="sequential_support",
            maximum=10**9,
        )
        absolute_error_sum = _finite_float(
            value.get("absolute_error_sum"),
            field="absolute_error_sum",
            minimum=0.0,
        )
        covered_values = _nonnegative_int(
            value.get("covered_values"),
            field="covered_values",
            maximum=10**12,
        )
        observed_values = _nonnegative_int(
            value.get("observed_values"),
            field="observed_values",
            maximum=10**12,
        )
        if covered_values > observed_values:
            raise ValueError("stored crystal coverage counts are inconsistent")
        return cls(
            key=key,
            feature_dim=feature_dim,
            support=support,
            feature_sum=feature_sum,
            target_sum=target_sum,
            gram=gram,
            cross=cross,
            residual_rows=residual_rows,
            sequential_support=sequential_support,
            absolute_error_sum=absolute_error_sum,
            covered_values=covered_values,
            observed_values=observed_values,
        )


class ForecastCrystalBank:
    """Centered ridge kernels with deterministic verification and persistence."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        ridge: float = 1.0,
        min_support: int = 8,
        min_sequential_support: int = 4,
        max_mean_absolute_error: float | None = None,
        min_coverage: float | None = None,
        quantile_levels: Sequence[object] = (0.1, 0.5, 0.9),
        max_residual_history: int = 2048,
    ) -> None:
        self._path = None if root is None else self._resolve_path(root)
        self._ridge = _finite_float(ridge, field="ridge", minimum=0.0)
        self._min_support = _positive_int(
            min_support,
            field="min_support",
            maximum=10**6,
        )
        self._min_sequential_support = _nonnegative_int(
            min_sequential_support,
            field="min_sequential_support",
            maximum=10**6,
        )
        if max_mean_absolute_error is not None:
            max_mean_absolute_error = _finite_float(
                max_mean_absolute_error,
                field="max_mean_absolute_error",
                minimum=0.0,
            )
        if min_coverage is not None:
            min_coverage = _finite_float(
                min_coverage,
                field="min_coverage",
                minimum=0.0,
            )
            if min_coverage > 1.0:
                raise ValueError("min_coverage must not exceed 1")
        self._max_mean_absolute_error = max_mean_absolute_error
        self._min_coverage = min_coverage
        self._quantile_levels = _quantile_levels(quantile_levels)
        self._max_residual_history = _positive_int(
            max_residual_history,
            field="max_residual_history",
            maximum=10**6,
        )
        self._states: dict[ForecastCrystalKey, _CrystalState] = {}
        if self._path is not None and self._path.exists():
            self._load()

    @staticmethod
    def _resolve_path(root: str | os.PathLike[str]) -> Path:
        path = Path(root).expanduser().absolute()
        if path.suffix == ".json":
            return path
        return path / "forecast_crystals.json"

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        return self._quantile_levels

    def describe(self, key: ForecastCrystalKey) -> ForecastCrystalStats | None:
        state = self._states.get(key)
        if state is None:
            return None
        return self._stats_for(state)

    def predict(
        self,
        key: ForecastCrystalKey,
        features: object,
        *,
        regime: str | None = None,
        quantile_levels: Sequence[object] | None = None,
    ) -> ForecastCrystalPrediction | None:
        if not isinstance(key, ForecastCrystalKey):
            raise TypeError("key must be ForecastCrystalKey")
        if regime is not None and _canonical_text(regime, field="regime") != key.regime:
            return None
        state = self._states.get(key)
        if state is None:
            return None
        stats = self._stats_for(state)
        if not stats.ready:
            return None
        levels = (
            self._quantile_levels
            if quantile_levels is None
            else _quantile_levels(quantile_levels)
        )
        vector = _feature_vector(features, expected_size=state.feature_dim)
        point_flat = self._predict_flat(state, vector)
        quantile_map = self._quantiles_for(state, point_flat, levels)
        shape = (key.horizon, key.target_count)
        return ForecastCrystalPrediction(
            key=key,
            point=_readonly_float32(point_flat.reshape(shape), shape=shape),
            quantiles=MappingProxyType(
                {
                    level: _readonly_float32(values.reshape(shape), shape=shape)
                    for level, values in quantile_map.items()
                }
            ),
            support=state.support,
            sequential_support=state.sequential_support,
            mean_absolute_error=stats.mean_absolute_error,
            empirical_coverage=stats.empirical_coverage,
        )

    def observe(
        self,
        key: ForecastCrystalKey,
        features: object,
        future_target: object,
        *,
        future_target_mask: object | None = None,
        fit_target: object | None = None,
        fit_target_mask: object | None = None,
        update_fit: bool = True,
        regime: str | None = None,
    ) -> ForecastCrystalStats:
        if not isinstance(key, ForecastCrystalKey):
            raise TypeError("key must be ForecastCrystalKey")
        if not isinstance(update_fit, bool):
            raise TypeError("update_fit must be boolean")
        if regime is not None and _canonical_text(regime, field="regime") != key.regime:
            raise ValueError("regime does not match the crystal key")
        state = self._states.get(key)
        if state is None:
            if not update_fit:
                raise ValueError(
                    "cannot score an unfitted crystal without a fit update"
                )
            vector = _feature_vector(features)
            state = _CrystalState(
                key=key,
                feature_dim=int(vector.size),
                support=0,
                feature_sum=np.zeros(int(vector.size), dtype=np.float64),
                target_sum=np.zeros(key.output_dim, dtype=np.float64),
                gram=np.zeros((int(vector.size), int(vector.size)), dtype=np.float64),
                cross=np.zeros((int(vector.size), key.output_dim), dtype=np.float64),
                residual_rows=[],
                sequential_support=0,
                absolute_error_sum=0.0,
                covered_values=0,
                observed_values=0,
            )
            self._states[key] = state
        vector = _feature_vector(features, expected_size=state.feature_dim)
        future = _future_matrix(
            future_target,
            horizon=key.horizon,
            target_count=key.target_count,
            mask=future_target_mask,
        )
        observed_mask = _future_mask(
            future_target_mask,
            horizon=key.horizon,
            target_count=key.target_count,
        )
        observed_flat = observed_mask.reshape(key.output_dim)
        observed_count = int(np.count_nonzero(observed_flat))
        actual_flat = future.reshape(key.output_dim)
        if fit_target is None:
            fit_future = future
            fit_mask = observed_mask
        else:
            fit_future = _future_matrix(
                fit_target,
                horizon=key.horizon,
                target_count=key.target_count,
                mask=fit_target_mask,
            )
            fit_mask = _future_mask(
                fit_target_mask,
                horizon=key.horizon,
                target_count=key.target_count,
            )
        fit_flat = fit_future.reshape(key.output_dim)
        fit_observed_flat = fit_mask.reshape(key.output_dim)
        if state.support > 0:
            point_before = self._predict_flat(state, vector)
            target_flat = np.where(fit_observed_flat, fit_flat, point_before)
        else:
            point_before = np.zeros(key.output_dim, dtype=np.float64)
            target_flat = np.where(fit_observed_flat, fit_flat, 0.0)

        if state.support > 0 and observed_count > 0:
            residual = np.where(observed_flat, actual_flat - point_before, 0.0)
            observed_residual = residual[observed_flat]
            state.sequential_support += 1
            state.absolute_error_sum += float(np.abs(observed_residual).sum())
            quantiles_before = self._quantiles_for(
                state, point_before, self._quantile_levels
            )
            lower = quantiles_before[min(quantiles_before)]
            upper = quantiles_before[max(quantiles_before)]
            state.covered_values += int(
                np.count_nonzero(
                    observed_flat & (actual_flat >= lower) & (actual_flat <= upper)
                )
            )
            state.observed_values += observed_count
            state.residual_rows.append([float(item) for item in residual.tolist()])
            if len(state.residual_rows) > self._max_residual_history:
                del state.residual_rows[
                    : len(state.residual_rows) - self._max_residual_history
                ]

        if update_fit:
            state.support += 1
            state.feature_sum = state.feature_sum + vector
            state.target_sum = state.target_sum + target_flat
            state.gram = state.gram + np.outer(vector, vector)
            state.cross = state.cross + np.outer(vector, target_flat)

        self._persist()
        return self._stats_for(state)

    def _predict_flat(self, state: _CrystalState, vector: FloatVector) -> FloatVector:
        count = float(state.support)
        if count <= 0.0:
            raise ValueError("crystal has no support")
        feature_mean = state.feature_sum / count
        target_mean = state.target_sum / count
        centered_gram = (
            state.gram - np.outer(state.feature_sum, state.feature_sum) / count
        )
        centered_cross = (
            state.cross - np.outer(state.feature_sum, state.target_sum) / count
        )
        centered_gram = 0.5 * (centered_gram + centered_gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(centered_gram)
        positive = np.maximum(eigenvalues, 0.0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            centered_gram = (eigenvectors * positive) @ eigenvectors.T
        scale = max(
            1.0,
            float(np.trace(centered_gram)) / max(1, state.feature_dim),
        )
        regularized = centered_gram + (
            self._ridge * scale * np.eye(state.feature_dim, dtype=np.float64)
        )
        try:
            coefficients = np.linalg.solve(regularized, centered_cross)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(regularized, rcond=1e-10) @ centered_cross
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            intercept = target_mean - (feature_mean @ coefficients)
            point = intercept + (vector @ coefficients)
        if point.shape != (state.key.output_dim,) or not bool(np.isfinite(point).all()):
            raise ValueError("crystal prediction became non-finite")
        return point

    def _quantiles_for(
        self,
        state: _CrystalState,
        point_flat: FloatVector,
        levels: Sequence[float],
    ) -> dict[float, FloatVector]:
        residual_rows = state.residual_rows
        if residual_rows:
            residual_matrix = np.array(residual_rows, dtype=np.float64, order="C")
            if (
                residual_matrix.ndim != 2
                or residual_matrix.shape[1] != state.key.output_dim
                or not bool(np.isfinite(residual_matrix).all())
            ):
                raise ValueError("stored crystal residuals are invalid")
        else:
            residual_matrix = np.zeros((1, state.key.output_dim), dtype=np.float64)
        quantiles: dict[float, FloatVector] = {}
        previous: FloatVector | None = None
        for level in levels:
            residual_quantile = np.quantile(
                residual_matrix,
                level,
                axis=0,
                method="linear",
            )
            values = point_flat + residual_quantile
            if previous is not None:
                values = np.maximum(values, previous)
            quantiles[level] = values
            previous = values
        return quantiles

    def _stats_for(self, state: _CrystalState) -> ForecastCrystalStats:
        mae = state.mean_absolute_error()
        coverage = state.empirical_coverage()
        ready = state.support >= self._min_support
        ready = ready and state.sequential_support >= self._min_sequential_support
        ready = ready and state.residual_support > 0
        if ready and self._max_mean_absolute_error is not None:
            ready = mae is not None and mae <= self._max_mean_absolute_error
        if ready and self._min_coverage is not None:
            ready = coverage is not None and coverage >= self._min_coverage
        return ForecastCrystalStats(
            key=state.key,
            feature_dim=state.feature_dim,
            support=state.support,
            sequential_support=state.sequential_support,
            residual_support=state.residual_support,
            mean_absolute_error=mae,
            empirical_coverage=coverage,
            ready=ready,
        )

    def _persist(self) -> None:
        if self._path is None:
            return
        _atomic_json_dump(
            self._path,
            {
                "entries": [state.to_document() for state in self._states.values()],
                "max_residual_history": self._max_residual_history,
                "min_coverage": self._min_coverage,
                "min_sequential_support": self._min_sequential_support,
                "min_support": self._min_support,
                "quantile_levels": list(self._quantile_levels),
                "ridge": self._ridge,
                "schema": _SCHEMA,
            },
        )

    def _load(self) -> None:
        assert self._path is not None
        document = _json_load(self._path)
        if document.get("schema") != _SCHEMA:
            raise ValueError("forecast crystal state schema is unsupported")
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise ValueError("forecast crystal state entries are invalid")
        states: dict[ForecastCrystalKey, _CrystalState] = {}
        for entry in entries:
            state = _CrystalState.from_document(entry)
            states[state.key] = state
        self._states = states


__all__ = [
    "ForecastCrystalBank",
    "ForecastCrystalKey",
    "ForecastCrystalPrediction",
    "ForecastCrystalStats",
]
