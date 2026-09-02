"""Structural zero-forward forecasting baselines."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from .contracts import ForecastInputs, ForecastPrediction


class SeasonalNaiveForecaster:
    def __init__(
        self,
        *,
        period: int = 24,
        quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> None:
        if period <= 0:
            raise ValueError("period must be positive")
        levels = tuple(float(level) for level in quantile_levels)
        if levels != tuple(sorted(set(levels))) or any(
            level <= 0.0 or level >= 1.0 for level in levels
        ):
            raise ValueError("quantile_levels must be sorted inside (0, 1)")
        self.period = period
        self.quantile_levels = levels

    def predict(self, inputs: ForecastInputs) -> ForecastPrediction:
        if not isinstance(inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        history = np.asarray(inputs.past_target, dtype=np.float64)
        mask = np.asarray(inputs.past_target_mask, dtype=np.bool_)
        point = np.empty((inputs.horizon, inputs.target_dim), dtype=np.float64)
        scales = np.empty(inputs.target_dim, dtype=np.float64)
        for target in range(inputs.target_dim):
            observed = history[mask[:, target], target]
            if observed.size == 0:
                raise ValueError("seasonal baseline needs one observed target")
            pattern = observed[-min(self.period, observed.size) :]
            point[:, target] = np.resize(pattern, inputs.horizon)
            if observed.size > self.period:
                residual = observed[self.period :] - observed[: -self.period]
            elif observed.size > 1:
                residual = np.diff(observed)
            else:
                residual = np.asarray([0.0])
            scales[target] = max(1e-4, float(np.std(residual)))
        quantiles = np.empty(
            (inputs.horizon, inputs.target_dim, len(self.quantile_levels)),
            dtype=np.float64,
        )
        for index, level in enumerate(self.quantile_levels):
            quantiles[..., index] = point + NormalDist().inv_cdf(level) * scales
        return ForecastPrediction(
            series_id=inputs.series_id,
            timestamps=inputs.future_timestamps,
            point=point.astype(np.float32),
            quantiles=quantiles.astype(np.float32),
            quantile_levels=self.quantile_levels,
        )


__all__ = ["SeasonalNaiveForecaster"]
