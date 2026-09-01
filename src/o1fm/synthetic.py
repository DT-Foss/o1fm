"""Deterministic multivariate forecasting regimes for tests and examples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .contracts import ForecastExample, ForecastInputs, ForecastTarget


def _readonly(value: NDArray[Any]) -> NDArray[Any]:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _positive_int(value: object, field: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
    ):
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _finite_non_negative(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Bounded controls for a reproducible multivariate forecasting corpus."""

    seed: int = 17
    n_series: int = 3
    total_steps: int = 240
    history_length: int = 48
    horizon: int = 12
    target_dim: int = 3
    past_covariate_dim: int = 2
    seasonal_period: int = 24
    train_fraction: float = 0.70
    train_stride: int = 6
    online_stride: int = 12
    intervention_offset: int = 4
    noise_scale: float = 0.04
    missing_rate: float = 0.03
    start_timestamp: str = "2024-01-01T00:00:00"
    step_minutes: int = 60

    def __post_init__(self) -> None:
        for field in (
            "n_series",
            "total_steps",
            "history_length",
            "horizon",
            "past_covariate_dim",
            "seasonal_period",
            "train_stride",
            "online_stride",
            "intervention_offset",
            "step_minutes",
        ):
            _positive_int(getattr(self, field), field)
        _positive_int(self.target_dim, "target_dim", minimum=2)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.train_fraction, bool)
            or not isinstance(self.train_fraction, (int, float))
            or not 0.0 < float(self.train_fraction) < 1.0
        ):
            raise ValueError("train_fraction must lie strictly between zero and one")
        _finite_non_negative(self.noise_scale, "noise_scale")
        missing_rate = _finite_non_negative(self.missing_rate, "missing_rate")
        if missing_rate >= 1.0:
            raise ValueError("missing_rate must be below one")
        if not isinstance(self.start_timestamp, str) or not self.start_timestamp:
            raise ValueError("start_timestamp must be non-empty text")
        try:
            start = np.datetime64(self.start_timestamp, "ns")
        except ValueError as exc:
            raise ValueError("start_timestamp is invalid") from exc
        if np.isnat(start):
            raise ValueError("start_timestamp cannot be missing")
        split = self.split_index
        if split < self.history_length + self.horizon:
            raise ValueError("training split is too short for one complete window")
        if self.total_steps - split < self.horizon:
            raise ValueError("online split is too short for one forecast horizon")
        if self.intervention_offset >= self.horizon:
            raise ValueError(
                "intervention_offset must lie inside the first online horizon"
            )
        first_switch, second_switch = self.regime_switch_indices
        if not 1 < first_switch < second_switch < self.total_steps - 1:
            raise ValueError("total_steps is too short for three regimes")

    @property
    def split_index(self) -> int:
        return int(self.total_steps * float(self.train_fraction))

    @property
    def intervention_index(self) -> int:
        return self.split_index + self.intervention_offset

    @property
    def regime_switch_indices(self) -> tuple[int, int]:
        return self.total_steps // 3, (2 * self.total_steps) // 3


@dataclass(frozen=True, slots=True)
class SyntheticSeries:
    """One complete latent series plus observation masks and regime labels."""

    series_id: str
    timestamps: NDArray[np.datetime64]
    target: NDArray[np.float32]
    target_mask: NDArray[np.bool_]
    past_covariates: NDArray[np.float32]
    past_covariate_mask: NDArray[np.bool_]
    known_covariates: NDArray[np.float32]
    known_covariate_mask: NDArray[np.bool_]
    regime_ids: NDArray[np.int8]

    def __post_init__(self) -> None:
        if not isinstance(self.series_id, str) or not self.series_id:
            raise ValueError("series_id must be non-empty")
        target = np.array(self.target, dtype=np.float32, copy=True, order="C")
        target_mask = np.array(
            self.target_mask,
            dtype=np.bool_,
            copy=True,
            order="C",
        )
        past = np.array(
            self.past_covariates,
            dtype=np.float32,
            copy=True,
            order="C",
        )
        past_mask = np.array(
            self.past_covariate_mask,
            dtype=np.bool_,
            copy=True,
            order="C",
        )
        known = np.array(
            self.known_covariates,
            dtype=np.float32,
            copy=True,
            order="C",
        )
        known_mask = np.array(
            self.known_covariate_mask,
            dtype=np.bool_,
            copy=True,
            order="C",
        )
        timestamps = np.array(
            self.timestamps,
            dtype="datetime64[ns]",
            copy=True,
            order="C",
        )
        regimes = np.array(self.regime_ids, dtype=np.int8, copy=True, order="C")
        steps = int(target.shape[0]) if target.ndim == 2 else -1
        if (
            steps < 1
            or target.shape[1] < 2
            or target_mask.shape != target.shape
            or past.ndim != 2
            or past.shape[0] != steps
            or past_mask.shape != past.shape
            or known.ndim != 2
            or known.shape[0] != steps
            or known_mask.shape != known.shape
            or timestamps.shape != (steps,)
            or regimes.shape != (steps,)
        ):
            raise ValueError("synthetic series arrays are misaligned")
        if not all(np.isfinite(value).all() for value in (target, past, known)):
            raise ValueError("synthetic series values must be finite")
        if np.isnat(timestamps).any() or (
            steps > 1 and (np.diff(timestamps.astype(np.int64)) <= 0).any()
        ):
            raise ValueError("synthetic timestamps must be strictly increasing")
        if not np.isin(regimes, (0, 1, 2)).all():
            raise ValueError("synthetic regime IDs must be 0, 1, or 2")
        target[~target_mask] = 0.0
        past[~past_mask] = 0.0
        known[~known_mask] = 0.0
        object.__setattr__(self, "timestamps", _readonly(timestamps))
        object.__setattr__(self, "target", _readonly(target))
        object.__setattr__(self, "target_mask", _readonly(target_mask))
        object.__setattr__(self, "past_covariates", _readonly(past))
        object.__setattr__(self, "past_covariate_mask", _readonly(past_mask))
        object.__setattr__(self, "known_covariates", _readonly(known))
        object.__setattr__(self, "known_covariate_mask", _readonly(known_mask))
        object.__setattr__(self, "regime_ids", _readonly(regimes))


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Complete series and leakage-safe train/online forecast windows."""

    config: SyntheticConfig
    series: tuple[SyntheticSeries, ...]
    train: tuple[ForecastExample, ...]
    online: tuple[ForecastExample, ...]
    split_index: int
    intervention_index: int
    regime_switch_indices: tuple[int, int]
    causal_covariate_lag: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.config, SyntheticConfig):
            raise TypeError("config must be SyntheticConfig")
        if (
            not self.series
            or len(self.series) != self.config.n_series
            or any(not isinstance(row, SyntheticSeries) for row in self.series)
        ):
            raise ValueError("series must contain every configured synthetic series")
        if not self.train or not self.online:
            raise ValueError("train and online splits must both be non-empty")
        if any(
            not isinstance(row, ForecastExample) for row in (*self.train, *self.online)
        ):
            raise TypeError("synthetic splits must contain ForecastExample values")
        if self.split_index != self.config.split_index:
            raise ValueError("dataset split index differs from its configuration")
        if self.intervention_index != self.config.intervention_index:
            raise ValueError(
                "dataset intervention index differs from its configuration"
            )
        if self.regime_switch_indices != self.config.regime_switch_indices:
            raise ValueError("dataset regime switches differ from its configuration")
        if self.causal_covariate_lag != 1:
            raise ValueError("synthetic causal covariates must use one past step")

    @property
    def target_dim(self) -> int:
        return self.config.target_dim

    @property
    def known_covariate_dim(self) -> int:
        return 4


def _timestamps(config: SyntheticConfig) -> NDArray[np.datetime64]:
    start = np.datetime64(config.start_timestamp, "ns")
    step = np.timedelta64(config.step_minutes, "m").astype("timedelta64[ns]")
    return start + np.arange(config.total_steps, dtype=np.int64) * step


def _known_covariates(config: SyntheticConfig) -> NDArray[np.float32]:
    time = np.arange(config.total_steps, dtype=np.float64)
    phase = 2.0 * np.pi * time / float(config.seasonal_period)
    intervention = (time >= config.intervention_index).astype(np.float64)
    trend = time / float(max(1, config.total_steps - 1))
    return np.stack((np.sin(phase), np.cos(phase), intervention, trend), axis=1).astype(
        np.float32
    )


def _regimes(config: SyntheticConfig) -> NDArray[np.int8]:
    first, second = config.regime_switch_indices
    result = np.zeros(config.total_steps, dtype=np.int8)
    result[first:second] = 1
    result[second:] = 2
    return result


def _transition_matrices(target_dim: int) -> tuple[NDArray[np.float64], ...]:
    identity = np.eye(target_dim, dtype=np.float64)
    forward = np.roll(identity, 1, axis=1)
    backward = np.roll(identity, -1, axis=1)
    return (
        0.52 * identity + 0.07 * forward,
        0.30 * identity - 0.08 * backward,
        0.62 * identity + 0.04 * forward - 0.03 * backward,
    )


def _make_series(
    config: SyntheticConfig,
    *,
    ordinal: int,
    rng: np.random.Generator,
) -> SyntheticSeries:
    steps = config.total_steps
    past_dim = config.past_covariate_dim
    target_dim = config.target_dim
    known = _known_covariates(config)
    regimes = _regimes(config)

    causal = np.zeros((steps, past_dim), dtype=np.float64)
    causal[0] = rng.normal(0.0, 0.35, size=past_dim)
    persistence = np.linspace(0.55, 0.82, past_dim, dtype=np.float64)
    for index in range(1, steps):
        innovation = rng.normal(0.0, 0.28, size=past_dim)
        causal[index] = persistence * causal[index - 1] + innovation

    transitions = _transition_matrices(target_dim)
    causal_weights = rng.normal(0.0, 0.16, size=(target_dim, past_dim))
    season_weights = rng.normal(0.0, 0.25, size=(target_dim, 2))
    intervention_effect = np.linspace(0.45, -0.25, target_dim, dtype=np.float64)
    regime_offsets = np.stack(
        (
            np.linspace(0.00, 0.10, target_dim),
            np.linspace(0.22, -0.12, target_dim),
            np.linspace(-0.18, 0.20, target_dim),
        )
    )
    target = np.zeros((steps, target_dim), dtype=np.float64)
    series_offset = (ordinal - (config.n_series - 1) / 2.0) * 0.08
    for index in range(steps):
        regime = int(regimes[index])
        previous_target = target[index - 1] if index else np.zeros(target_dim)
        previous_causal = causal[index - 1] if index else np.zeros(past_dim)
        seasonal = season_weights @ known[index, :2].astype(np.float64)
        intervention = intervention_effect * float(known[index, 2])
        noise = rng.normal(0.0, config.noise_scale, size=target_dim)
        target[index] = (
            transitions[regime] @ previous_target
            + causal_weights @ previous_causal
            + seasonal
            + intervention
            + regime_offsets[regime]
            + series_offset
            + noise
        )

    target_mask = rng.random(target.shape) >= config.missing_rate
    causal_mask = rng.random(causal.shape) >= config.missing_rate
    target_mask[0] = True
    causal_mask[0] = True
    target_values = target.astype(np.float32)
    causal_values = causal.astype(np.float32)
    target_values[~target_mask] = 0.0
    causal_values[~causal_mask] = 0.0
    known_mask = np.ones(known.shape, dtype=np.bool_)
    return SyntheticSeries(
        series_id=f"series-{ordinal:03d}",
        timestamps=_timestamps(config),
        target=target_values,
        target_mask=target_mask,
        past_covariates=causal_values,
        past_covariate_mask=causal_mask,
        known_covariates=known,
        known_covariate_mask=known_mask,
        regime_ids=regimes,
    )


def _window(
    series: SyntheticSeries,
    config: SyntheticConfig,
    *,
    origin: int,
) -> ForecastExample:
    history_start = origin - config.history_length
    future_end = origin + config.horizon
    inputs = ForecastInputs(
        series_id=series.series_id,
        horizon=config.horizon,
        timestamps=series.timestamps[history_start:future_end],
        past_target=series.target[history_start:origin],
        past_target_mask=series.target_mask[history_start:origin],
        past_covariates=series.past_covariates[history_start:origin],
        past_covariate_mask=series.past_covariate_mask[history_start:origin],
        known_covariates=series.known_covariates[history_start:future_end],
        known_covariate_mask=series.known_covariate_mask[history_start:future_end],
    )
    target = ForecastTarget(
        series_id=series.series_id,
        timestamps=series.timestamps[origin:future_end],
        values=series.target[origin:future_end],
        mask=series.target_mask[origin:future_end],
    )
    return ForecastExample(inputs=inputs, target=target)


def generate_synthetic_dataset(
    config: SyntheticConfig | None = None,
) -> SyntheticDataset:
    """Generate deterministic regimes and aligned train/online windows.

    Past-only covariates affect targets with lag one.  Sine/cosine seasonality,
    a trend, and a scheduled intervention are known through the future horizon.
    Regime switches alter the stable multivariate autoregression twice.
    """

    selected = SyntheticConfig() if config is None else config
    if not isinstance(selected, SyntheticConfig):
        raise TypeError("config must be SyntheticConfig or None")
    rng = np.random.default_rng(selected.seed)
    series = tuple(
        _make_series(selected, ordinal=index, rng=rng)
        for index in range(selected.n_series)
    )
    train_origins = range(
        selected.history_length,
        selected.split_index - selected.horizon + 1,
        selected.train_stride,
    )
    online_origins = range(
        selected.split_index,
        selected.total_steps - selected.horizon + 1,
        selected.online_stride,
    )
    train = tuple(
        _window(row, selected, origin=origin)
        for row in series
        for origin in train_origins
    )
    online = tuple(
        _window(row, selected, origin=origin)
        for row in series
        for origin in online_origins
    )
    return SyntheticDataset(
        config=selected,
        series=series,
        train=train,
        online=online,
        split_index=selected.split_index,
        intervention_index=selected.intervention_index,
        regime_switch_indices=selected.regime_switch_indices,
    )


make_synthetic_dataset = generate_synthetic_dataset


__all__ = [
    "SyntheticConfig",
    "SyntheticDataset",
    "SyntheticSeries",
    "generate_synthetic_dataset",
    "make_synthetic_dataset",
]
