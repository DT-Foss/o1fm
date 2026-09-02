"""Fixed forecast-context features used by local Forecast Crystals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .contracts import ForecastInputs

FEATURE_ABI = "o1fm.level-slope-seasonal-known-future/v1"


def _selected(count: int, selected: Sequence[int] | None) -> tuple[int, ...]:
    if selected is None:
        return tuple(range(count))
    result = tuple(int(value) for value in selected)
    if result != tuple(sorted(set(result))) or any(
        value < 0 or value >= count for value in result
    ):
        raise ValueError("selected feature indices are invalid")
    return result


def _column_features(
    values: NDArray[np.float32],
    mask: NDArray[np.bool_],
    *,
    seasonal_period: int,
) -> list[float]:
    observed = values[mask].astype(np.float64, copy=False)
    if observed.size == 0:
        return [0.0] * 7
    tail = observed[-min(8, observed.size) :]
    x = np.arange(tail.size, dtype=np.float64)
    slope = 0.0
    if tail.size > 1:
        centered = x - x.mean()
        slope = float(np.dot(centered, tail - tail.mean()) / np.dot(centered, centered))
    seasonal = (
        float(observed[-seasonal_period])
        if observed.size >= seasonal_period
        else float(observed[0])
    )
    return [
        float(observed[-1]),
        float(observed.mean()),
        float(observed.std()),
        slope,
        seasonal,
        float(observed[-1] - seasonal),
        float(observed.size / max(1, values.size)),
    ]


def feature_schema(
    inputs: ForecastInputs,
    *,
    selected_past: Sequence[int] | None = None,
    selected_known: Sequence[int] | None = None,
    seasonal_period: int = 24,
) -> str:
    past = _selected(inputs.past_covariate_dim, selected_past)
    known = _selected(inputs.known_covariate_dim, selected_known)
    record = {
        "abi": FEATURE_ABI,
        "known": list(known),
        "past": list(past),
        "seasonal_period": int(seasonal_period),
        "target_dim": inputs.target_dim,
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_features(
    inputs: ForecastInputs,
    *,
    selected_past: Sequence[int] | None = None,
    selected_known: Sequence[int] | None = None,
    seasonal_period: int = 24,
) -> NDArray[np.float64]:
    """Encode levels, trends, seasonal state and known-future interventions."""

    if not isinstance(inputs, ForecastInputs):
        raise TypeError("inputs must be ForecastInputs")
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive")
    past = _selected(inputs.past_covariate_dim, selected_past)
    known = _selected(inputs.known_covariate_dim, selected_known)
    rows: list[float] = [1.0, float(inputs.horizon)]
    for index in range(inputs.target_dim):
        rows.extend(
            _column_features(
                inputs.past_target[:, index],
                inputs.past_target_mask[:, index],
                seasonal_period=seasonal_period,
            )
        )
    for index in past:
        rows.extend(
            _column_features(
                inputs.past_covariates[:, index],
                inputs.past_covariate_mask[:, index],
                seasonal_period=seasonal_period,
            )
        )
    history = inputs.history_length
    for index in known:
        context = inputs.known_covariates[:history, index]
        context_mask = inputs.known_covariate_mask[:history, index]
        future = inputs.known_covariates[history:, index]
        future_mask = inputs.known_covariate_mask[history:, index]
        rows.extend(
            _column_features(
                context,
                context_mask,
                seasonal_period=seasonal_period,
            )
        )
        observed_future = future[future_mask].astype(np.float64, copy=False)
        if observed_future.size:
            rows.extend(
                [
                    float(observed_future[0]),
                    float(observed_future[-1]),
                    float(observed_future.mean()),
                    float(observed_future.min()),
                    float(observed_future.max()),
                ]
            )
        else:
            rows.extend([0.0] * 5)
    result = np.asarray(rows, dtype=np.float64)
    if not bool(np.isfinite(result).all()):
        raise ValueError("forecast features are not finite")
    result.setflags(write=False)
    return result


__all__ = ["FEATURE_ABI", "extract_features", "feature_schema"]
