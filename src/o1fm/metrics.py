"""Forecast verification and proper sequential scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contracts import ForecastPrediction, ForecastTarget


@dataclass(frozen=True, slots=True)
class ForecastScore:
    mae: float
    rmse: float
    pinball: float | None
    interval_coverage: float | None
    interval_width: float | None
    observed_values: int

    @property
    def quality(self) -> float:
        """Lower is better; probabilistic forecasts use proper pinball loss."""

        return self.mae if self.pinball is None else self.pinball


def verify_prediction(
    prediction: ForecastPrediction,
    target: ForecastTarget | None = None,
) -> None:
    """Enforce the complete public forecast contract."""

    if not isinstance(prediction, ForecastPrediction):
        raise TypeError("prediction must be ForecastPrediction")
    if not bool(np.isfinite(prediction.point).all()):
        raise ValueError("point forecast must be finite")
    if prediction.quantiles is not None:
        if not bool(np.isfinite(prediction.quantiles).all()):
            raise ValueError("quantile forecasts must be finite")
        if bool((np.diff(prediction.quantiles, axis=2) < 0.0).any()):
            raise ValueError("quantile forecasts must be non-crossing")
    if target is not None:
        if not isinstance(target, ForecastTarget):
            raise TypeError("target must be ForecastTarget or None")
        if (
            prediction.series_id != target.series_id
            or prediction.horizon != target.horizon
            or prediction.target_dim != target.target_dim
            or not np.array_equal(prediction.timestamps, target.timestamps)
        ):
            raise ValueError("prediction and realized target contracts differ")


def score_prediction(
    prediction: ForecastPrediction,
    target: ForecastTarget,
) -> ForecastScore:
    """Score one settled forecast using only observed future values."""

    verify_prediction(prediction, target)
    observed = np.asarray(target.mask, dtype=np.bool_)
    count = int(observed.sum())
    if count <= 0:
        raise ValueError("settlement requires at least one observed target")
    residual = np.asarray(target.values - prediction.point, dtype=np.float64)
    selected = residual[observed]
    mae = float(np.abs(selected).mean())
    rmse = float(np.sqrt(np.square(selected).mean()))
    pinball: float | None = None
    coverage: float | None = None
    width: float | None = None
    if prediction.quantiles is not None:
        assert prediction.quantile_levels is not None
        levels = np.asarray(prediction.quantile_levels, dtype=np.float64)
        actual = np.asarray(target.values, dtype=np.float64)[..., None]
        error = actual - np.asarray(prediction.quantiles, dtype=np.float64)
        loss = np.maximum(levels * error, (levels - 1.0) * error)
        pinball = float(loss[observed].mean())
        lower = prediction.quantiles[..., 0]
        upper = prediction.quantiles[..., -1]
        coverage = float(
            (
                (target.values[observed] >= lower[observed])
                & (target.values[observed] <= upper[observed])
            ).mean()
        )
        width = float((upper[observed] - lower[observed]).mean())
    values = (mae, rmse, pinball, coverage, width)
    if any(value is not None and not math.isfinite(value) for value in values):
        raise ValueError("forecast score is not finite")
    return ForecastScore(
        mae=mae,
        rmse=rmse,
        pinball=pinball,
        interval_coverage=coverage,
        interval_width=width,
        observed_values=count,
    )


__all__ = ["ForecastScore", "score_prediction", "verify_prediction"]
