"""Strict, leakage-resistant contracts for multivariate forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

Float32Matrix = NDArray[np.float32]
BoolMatrix = NDArray[np.bool_]
TimestampVector = NDArray[np.datetime64]

_MAX_HORIZON = 1_000_000
_MAX_SERIES_ID_BYTES = 1024


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= maximum
    ):
        raise ValueError(f"{field} must be a positive bounded integer")
    return int(value)


def _series_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_SERIES_ID_BYTES
    ):
        raise ValueError("series_id must be canonical non-empty text")
    return value


def _readonly(value: NDArray[Any]) -> NDArray[Any]:
    value.setflags(write=False)
    return value


def _boolean_mask(
    value: object | None,
    *,
    shape: tuple[int, int],
    field: str,
) -> BoolMatrix:
    if value is None:
        return _readonly(np.ones(shape, dtype=np.bool_, order="C"))
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_) or raw.ndim != 2 or raw.shape != shape:
        raise ValueError(f"{field} must be a boolean matrix with shape {shape}")
    return _readonly(np.array(raw, dtype=np.bool_, order="C", copy=True))


def _float32_matrix(
    value: object,
    *,
    mask: object | None,
    field: str,
    rows: int | None = None,
    allow_zero_columns: bool = False,
) -> tuple[Float32Matrix, BoolMatrix]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{field} cannot contain complex values")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.array(value, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{field} must be a rank-two [time, feature] matrix")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{field} must contain exactly {rows} time rows")
    if matrix.shape[0] < 1:
        raise ValueError(f"{field} must contain at least one time row")
    if matrix.shape[1] < int(not allow_zero_columns):
        raise ValueError(f"{field} must contain at least one feature column")
    observed = _boolean_mask(mask, shape=matrix.shape, field=f"{field}_mask")
    if bool(np.isinf(matrix).any()):
        raise ValueError(f"{field} cannot contain infinity")
    if bool((observed & ~np.isfinite(matrix)).any()):
        raise ValueError(f"{field} has non-finite values marked as observed")
    matrix[~observed] = np.float32(0.0)
    matrix[matrix == 0.0] = np.float32(0.0)
    if not bool(np.isfinite(matrix).all()):  # Defensive canonicalization check.
        raise ValueError(f"{field} could not be canonicalized to finite float32")
    return _readonly(matrix), observed


def _empty_covariates(rows: int) -> tuple[Float32Matrix, BoolMatrix]:
    values = np.empty((rows, 0), dtype=np.float32, order="C")
    mask = np.empty((rows, 0), dtype=np.bool_, order="C")
    return _readonly(values), _readonly(mask)


def _float32_quantiles(
    value: object,
    *,
    shape_prefix: tuple[int, int],
) -> NDArray[np.float32]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("quantiles cannot contain complex values")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            quantiles = np.array(
                value,
                dtype=np.float32,
                order="C",
                copy=True,
            )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("quantiles must be numeric") from exc
    if (
        quantiles.ndim != 3
        or quantiles.shape[:2] != shape_prefix
        or quantiles.shape[2] < 1
    ):
        raise ValueError("quantiles must have shape [horizon, target, quantile_level]")
    if not bool(np.isfinite(quantiles).all()):
        raise ValueError("quantiles must contain only finite values")
    if quantiles.shape[2] > 1 and bool((np.diff(quantiles, axis=2) < 0.0).any()):
        raise ValueError("quantile forecasts must be non-decreasing across levels")
    quantiles[quantiles == 0.0] = np.float32(0.0)
    return _readonly(quantiles)


def _quantile_levels(
    value: object,
    *,
    count: int,
) -> NDArray[np.float32]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("quantile_levels cannot contain complex values")
    try:
        levels = np.array(value, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("quantile_levels must be numeric") from exc
    if levels.ndim != 1 or levels.shape != (count,):
        raise ValueError(f"quantile_levels must contain exactly {count} values")
    if (
        not bool(np.isfinite(levels).all())
        or bool((levels <= 0.0).any())
        or bool((levels >= 1.0).any())
        or (count > 1 and bool((np.diff(levels) <= 0.0).any()))
    ):
        raise ValueError(
            "quantile_levels must be finite, strictly ordered, and inside (0, 1)"
        )
    return _readonly(levels)


def _covariates(
    value: object | None,
    *,
    mask: object | None,
    rows: int,
    field: str,
) -> tuple[Float32Matrix, BoolMatrix]:
    if value is None:
        if mask is not None:
            raise ValueError(f"{field}_mask requires {field}")
        return _empty_covariates(rows)
    return _float32_matrix(
        value,
        mask=mask,
        field=field,
        rows=rows,
        allow_zero_columns=True,
    )


def _timestamps(
    value: object,
    *,
    field: str,
    length: int,
) -> TimestampVector:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a timestamp sequence")
    try:
        raw = np.asarray(value)
        timestamps = np.array(value, dtype="datetime64[ns]", copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must contain valid timestamps") from exc
    if raw.ndim != 1 or timestamps.ndim != 1 or timestamps.shape != (length,):
        raise ValueError(f"{field} must contain exactly {length} timestamps")
    if bool(np.isnat(timestamps).any()):
        raise ValueError(f"{field} cannot contain missing timestamps")
    ticks = timestamps.astype(np.int64, copy=False)
    if length > 1 and bool((np.diff(ticks) <= 0).any()):
        raise ValueError(f"{field} must be strictly increasing and unique")
    return _readonly(timestamps)


@dataclass(frozen=True, slots=True)
class ForecastInputs:
    """Everything a forecasting model may observe at one forecast origin.

    ``past_target`` and ``past_covariates`` end at the forecast origin.  The
    known-covariate and timestamp axes include the same history followed by
    exactly ``horizon`` future rows.  Future target values have no field in
    this class, making leakage structurally impossible through model inputs.
    """

    series_id: str
    horizon: int
    timestamps: Sequence[object] | NDArray[Any]
    past_target: object
    past_target_mask: object | None = None
    past_covariates: object | None = None
    past_covariate_mask: object | None = None
    known_covariates: object | None = None
    known_covariate_mask: object | None = None

    def __post_init__(self) -> None:
        series_id = _series_id(self.series_id)
        horizon = _positive_int(
            self.horizon,
            field="horizon",
            maximum=_MAX_HORIZON,
        )
        target, target_mask = _float32_matrix(
            self.past_target,
            mask=self.past_target_mask,
            field="past_target",
        )
        history = int(target.shape[0])
        timeline = _timestamps(
            self.timestamps,
            field="timestamps",
            length=history + horizon,
        )
        past_covariates, past_covariate_mask = _covariates(
            self.past_covariates,
            mask=self.past_covariate_mask,
            rows=history,
            field="past_covariates",
        )
        known_covariates, known_covariate_mask = _covariates(
            self.known_covariates,
            mask=self.known_covariate_mask,
            rows=history + horizon,
            field="known_covariates",
        )
        object.__setattr__(self, "series_id", series_id)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "timestamps", timeline)
        object.__setattr__(self, "past_target", target)
        object.__setattr__(self, "past_target_mask", target_mask)
        object.__setattr__(self, "past_covariates", past_covariates)
        object.__setattr__(self, "past_covariate_mask", past_covariate_mask)
        object.__setattr__(self, "known_covariates", known_covariates)
        object.__setattr__(self, "known_covariate_mask", known_covariate_mask)

    @property
    def history_length(self) -> int:
        return int(self.past_target.shape[0])

    @property
    def target_dim(self) -> int:
        return int(self.past_target.shape[1])

    @property
    def past_covariate_dim(self) -> int:
        return int(self.past_covariates.shape[1])

    @property
    def known_covariate_dim(self) -> int:
        return int(self.known_covariates.shape[1])

    @property
    def past_timestamps(self) -> TimestampVector:
        return self.timestamps[: self.history_length]

    @property
    def future_timestamps(self) -> TimestampVector:
        return self.timestamps[self.history_length :]

    def model_inputs(self) -> Mapping[str, object]:
        """Return a read-only mapping with no future-target channel."""

        return MappingProxyType(
            {
                "horizon": self.horizon,
                "known_covariate_mask": self.known_covariate_mask,
                "known_covariates": self.known_covariates,
                "past_covariate_mask": self.past_covariate_mask,
                "past_covariates": self.past_covariates,
                "past_target": self.past_target,
                "past_target_mask": self.past_target_mask,
                "series_id": self.series_id,
                "timestamps": self.timestamps,
            }
        )


@dataclass(frozen=True, slots=True)
class ForecastTarget:
    """Future labels kept separate from model-visible forecasting inputs."""

    series_id: str
    timestamps: Sequence[object] | NDArray[Any]
    values: object
    mask: object | None = None

    def __post_init__(self) -> None:
        series_id = _series_id(self.series_id)
        values, mask = _float32_matrix(
            self.values,
            mask=self.mask,
            field="future_target",
        )
        timestamps = _timestamps(
            self.timestamps,
            field="future_target_timestamps",
            length=int(values.shape[0]),
        )
        object.__setattr__(self, "series_id", series_id)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "mask", mask)

    @property
    def horizon(self) -> int:
        return int(self.values.shape[0])

    @property
    def target_dim(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class ForecastExample:
    """One input window and its optional, separately aligned future labels."""

    inputs: ForecastInputs
    target: ForecastTarget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        if self.target is None:
            return
        if not isinstance(self.target, ForecastTarget):
            raise TypeError("target must be ForecastTarget or None")
        if self.target.series_id != self.inputs.series_id:
            raise ValueError("input and target series_id differ")
        if self.target.horizon != self.inputs.horizon:
            raise ValueError("target horizon differs from forecast horizon")
        if self.target.target_dim != self.inputs.target_dim:
            raise ValueError("past and future target dimensions differ")
        if not np.array_equal(self.target.timestamps, self.inputs.future_timestamps):
            raise ValueError(
                "future target timestamps differ from the forecast horizon"
            )

    @property
    def series_id(self) -> str:
        return self.inputs.series_id

    def require_target(self) -> ForecastTarget:
        if self.target is None:
            raise ValueError("forecast example has no future labels")
        return self.target


@dataclass(frozen=True, slots=True)
class ForecastPrediction:
    """Point and optional calibrated quantile forecasts for one series."""

    series_id: str
    timestamps: Sequence[object] | NDArray[Any]
    point: object
    quantiles: object | None = None
    quantile_levels: object | None = None

    def __post_init__(self) -> None:
        series_id = _series_id(self.series_id)
        point, _mask = _float32_matrix(
            self.point,
            mask=None,
            field="point",
        )
        timestamps = _timestamps(
            self.timestamps,
            field="prediction_timestamps",
            length=int(point.shape[0]),
        )
        if (self.quantiles is None) != (self.quantile_levels is None):
            raise ValueError("quantiles and quantile_levels must be supplied together")
        if self.quantiles is None:
            quantiles = None
            levels = None
        else:
            quantiles = _float32_quantiles(
                self.quantiles,
                shape_prefix=(int(point.shape[0]), int(point.shape[1])),
            )
            levels = _quantile_levels(
                self.quantile_levels,
                count=int(quantiles.shape[2]),
            )
        object.__setattr__(self, "series_id", series_id)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "quantile_levels", levels)

    @property
    def horizon(self) -> int:
        return int(self.point.shape[0])

    @property
    def target_dim(self) -> int:
        return int(self.point.shape[1])

    def validate_for(self, inputs: ForecastInputs) -> None:
        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        if self.series_id != inputs.series_id:
            raise ValueError("prediction and input series_id differ")
        if self.horizon != inputs.horizon or self.target_dim != inputs.target_dim:
            raise ValueError("prediction shape differs from the forecast contract")
        if not np.array_equal(self.timestamps, inputs.future_timestamps):
            raise ValueError("prediction timestamps differ from the forecast horizon")


# Compatibility spelling for callers that prefer the plural role name.
ForecastTargets = ForecastTarget
ForecastOutput = ForecastPrediction


__all__ = [
    "BoolMatrix",
    "Float32Matrix",
    "ForecastExample",
    "ForecastInputs",
    "ForecastOutput",
    "ForecastPrediction",
    "ForecastTarget",
    "ForecastTargets",
    "TimestampVector",
]
