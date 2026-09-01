from __future__ import annotations

import numpy as np
import pytest

from o1fm.contracts import ForecastExample
from o1fm.synthetic import SyntheticConfig, generate_synthetic_dataset


def _config(*, seed: int = 23) -> SyntheticConfig:
    return SyntheticConfig(
        seed=seed,
        n_series=2,
        total_steps=96,
        history_length=24,
        horizon=8,
        target_dim=3,
        past_covariate_dim=2,
        seasonal_period=12,
        train_fraction=0.70,
        train_stride=4,
        online_stride=8,
        intervention_offset=3,
        missing_rate=0.10,
    )


def test_generation_is_bitwise_deterministic_and_seed_sensitive() -> None:
    first = generate_synthetic_dataset(_config())
    second = generate_synthetic_dataset(_config())
    different = generate_synthetic_dataset(_config(seed=24))

    assert len(first.series) == len(second.series) == 2
    for left, right in zip(first.series, second.series, strict=True):
        for field in (
            "timestamps",
            "target",
            "target_mask",
            "past_covariates",
            "past_covariate_mask",
            "known_covariates",
            "known_covariate_mask",
            "regime_ids",
        ):
            assert np.array_equal(getattr(left, field), getattr(right, field))
    assert not np.array_equal(first.series[0].target, different.series[0].target)


def test_series_contains_three_regimes_seasonality_and_known_intervention() -> None:
    dataset = generate_synthetic_dataset(_config())
    series = dataset.series[0]
    first_switch, second_switch = dataset.regime_switch_indices

    assert np.array_equal(np.unique(series.regime_ids), np.array([0, 1, 2]))
    assert np.all(series.regime_ids[:first_switch] == 0)
    assert np.all(series.regime_ids[first_switch:second_switch] == 1)
    assert np.all(series.regime_ids[second_switch:] == 2)
    seasonal_norm = np.square(series.known_covariates[:, :2]).sum(axis=1)
    assert np.allclose(seasonal_norm, 1.0, atol=2e-7, rtol=0.0)
    intervention = series.known_covariates[:, 2]
    assert np.all(intervention[: dataset.intervention_index] == 0.0)
    assert np.all(intervention[dataset.intervention_index :] == 1.0)
    assert dataset.causal_covariate_lag == 1


def test_train_and_online_windows_are_aligned_without_future_target_leakage() -> None:
    dataset = generate_synthetic_dataset(_config())
    split_timestamp = dataset.series[0].timestamps[dataset.split_index]

    assert dataset.train
    assert dataset.online
    assert all(isinstance(row, ForecastExample) for row in dataset.train)
    for example in dataset.train:
        target = example.require_target()
        assert target.timestamps[-1] < split_timestamp
    for example in dataset.online:
        inputs = example.inputs
        target = example.require_target()
        assert inputs.future_timestamps[0] >= split_timestamp
        assert np.array_equal(target.timestamps, inputs.future_timestamps)
        assert inputs.past_target.shape == (24, 3)
        assert inputs.past_covariates.shape == (24, 2)
        assert inputs.known_covariates.shape == (32, 4)
        assert target.values.shape == (8, 3)
        assert not np.shares_memory(inputs.past_target, target.values)
        assert not np.shares_memory(inputs.known_covariates, target.values)
        assert "future_target" not in inputs.model_inputs()

    first_online = dataset.online[0].inputs
    future_intervention = first_online.known_covariates[
        first_online.history_length :, 2
    ]
    assert np.array_equal(
        future_intervention,
        np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float32),
    )


def test_masks_are_deterministic_finite_and_canonical() -> None:
    dataset = generate_synthetic_dataset(_config())
    series = dataset.series[0]

    assert (~series.target_mask).any()
    assert (~series.past_covariate_mask).any()
    assert np.isfinite(series.target).all()
    assert np.isfinite(series.past_covariates).all()
    assert np.all(series.target[~series.target_mask] == 0.0)
    assert np.all(series.past_covariates[~series.past_covariate_mask] == 0.0)
    assert not series.target.flags.writeable
    assert not dataset.train[0].inputs.past_target.flags.writeable
    assert not dataset.online[0].require_target().values.flags.writeable


@pytest.mark.parametrize(
    "changes",
    (
        {"target_dim": 1},
        {"horizon": 8, "intervention_offset": 8},
        {"missing_rate": 1.0},
        {"train_fraction": 0.1},
        {"total_steps": 40, "history_length": 24, "horizon": 12},
        {"noise_scale": float("nan")},
    ),
)
def test_invalid_synthetic_configurations_fail_closed(
    changes: dict[str, object],
) -> None:
    options: dict[str, object] = {
        "total_steps": 96,
        "history_length": 24,
        "horizon": 8,
        "intervention_offset": 3,
    }
    options.update(changes)
    with pytest.raises(ValueError):
        SyntheticConfig(**options)
