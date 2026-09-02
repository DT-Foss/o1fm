from __future__ import annotations

import hashlib

import numpy as np
import pytest

from o1fm.branches import InterventionPlan, InterventionSpec, apply_intervention
from o1fm.contracts import ForecastInputs
from o1fm.synthetic import (
    SyntheticConfig,
    SyntheticDataset,
    generate_synthetic_dataset,
)


def _config() -> SyntheticConfig:
    return SyntheticConfig(
        seed=23,
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


def _factual_digest(dataset: SyntheticDataset) -> str:
    digest = hashlib.sha256()
    for series in dataset.series:
        for field in (
            "target",
            "target_mask",
            "past_covariates",
            "past_covariate_mask",
            "known_covariates",
            "known_covariate_mask",
            "regime_ids",
        ):
            digest.update(getattr(series, field).tobytes())
    return digest.hexdigest()


def test_structural_state_preserves_factual_bytes_and_replays_every_window() -> None:
    dataset = generate_synthetic_dataset(_config())
    assert _factual_digest(dataset) == (
        "56f474f6a37fcd4dd735c01f6df76c2da17171f28aa0c273f99086ddba7855b4"
    )

    for example in (*dataset.train, *dataset.online):
        factual = example.require_target()
        replay = dataset.counterfactual_target(example.inputs)
        assert np.array_equal(replay.timestamps, factual.timestamps)
        assert np.array_equal(replay.mask, factual.mask)
        assert np.array_equal(replay.values, factual.values)
        assert replay.values.tobytes() == factual.values.tobytes()

    state = dataset.series[0].structural_state
    assert state.latent_target.shape == (96, 3)
    assert state.causal_process.shape == (96, 2)
    assert state.noise.shape == (96, 3)
    assert len(state.transition_matrices) == 3
    assert not state.latent_target.flags.writeable
    assert not state.noise.flags.writeable
    assert "future_target" not in dataset.online[0].inputs.model_inputs()


def test_intervention_uses_the_same_noise_path_and_propagates_exactly() -> None:
    dataset = generate_synthetic_dataset(_config())
    example = dataset.online[0]
    inputs = example.inputs
    factual = example.require_target()
    plan = InterventionPlan(
        "cancel-scheduled-intervention",
        (
            InterventionSpec(
                name="intervention-off",
                operation="set",
                timestamps=inputs.future_timestamps,
                covariate_indices=(2,),
                values=np.zeros((inputs.horizon, 1), dtype=np.float32),
                lower_bounds=(0.0,),
                upper_bounds=(1.0,),
            ),
        ),
    )
    edited = apply_intervention(inputs, plan)
    paired = dataset.counterfactual_target(edited)
    replay = dataset.counterfactual_target(edited)

    assert np.array_equal(paired.timestamps, factual.timestamps)
    assert np.array_equal(paired.mask, factual.mask)
    assert np.array_equal(paired.values, replay.values)
    assert not np.array_equal(paired.values, factual.values)
    np.testing.assert_array_equal(
        inputs.known_covariates[inputs.history_length :, 2],
        np.asarray((0, 0, 0, 1, 1, 1, 1, 1), dtype=np.float32),
    )

    series = dataset.series[0]
    state = series.structural_state
    origin = dataset.split_index
    expected = np.empty_like(state.latent_target[origin : origin + inputs.horizon])
    previous = state.latent_target[origin - 1]
    future_known = edited.known_covariates[edited.history_length :]
    for local_index in range(inputs.horizon):
        world_index = origin + local_index
        regime = int(state.regime_ids[world_index])
        known = future_known[local_index]
        current = (
            state.transition_matrices[regime] @ previous
            + state.causal_weights @ state.causal_process[world_index - 1]
            + state.season_weights @ known[:2].astype(np.float64)
            + state.intervention_effect * float(known[2])
            + state.regime_offsets[regime]
            + state.series_offset
            + state.noise[world_index]
        )
        expected[local_index] = current
        previous = current
    expected_values = expected.astype(np.float32)
    expected_values[~factual.mask] = 0.0
    np.testing.assert_array_equal(paired.values, expected_values)


def test_noncausal_known_channel_has_zero_paired_effect() -> None:
    dataset = generate_synthetic_dataset(_config())
    example = dataset.online[0]
    inputs = example.inputs
    factual = example.require_target()
    edit = InterventionPlan(
        "unused-trend-channel",
        (
            InterventionSpec(
                name="arbitrary-trend",
                operation="set",
                timestamps=inputs.future_timestamps,
                covariate_indices=(3,),
                values=np.full((inputs.horizon, 1), 99.0, dtype=np.float32),
            ),
        ),
    )
    paired = dataset.counterfactual_target(apply_intervention(inputs, edit))
    np.testing.assert_array_equal(paired.values, factual.values)


def test_oracle_rejects_unknown_worlds_history_edits_and_missing_schedules() -> None:
    dataset = generate_synthetic_dataset(_config())
    inputs = dataset.online[0].inputs
    unknown = ForecastInputs(
        series_id="unknown",
        horizon=inputs.horizon,
        timestamps=inputs.timestamps,
        past_target=inputs.past_target,
        past_target_mask=inputs.past_target_mask,
        past_covariates=inputs.past_covariates,
        past_covariate_mask=inputs.past_covariate_mask,
        known_covariates=inputs.known_covariates,
        known_covariate_mask=inputs.known_covariate_mask,
    )
    with pytest.raises(KeyError, match="absent"):
        dataset.counterfactual_target(unknown)

    changed_history = np.array(inputs.past_target, copy=True)
    changed_history[0, 0] += 1.0
    invalid_history = ForecastInputs(
        series_id=inputs.series_id,
        horizon=inputs.horizon,
        timestamps=inputs.timestamps,
        past_target=changed_history,
        past_target_mask=inputs.past_target_mask,
        past_covariates=inputs.past_covariates,
        past_covariate_mask=inputs.past_covariate_mask,
        known_covariates=inputs.known_covariates,
        known_covariate_mask=inputs.known_covariate_mask,
    )
    with pytest.raises(ValueError, match="only known-future"):
        dataset.counterfactual_target(invalid_history)

    missing_mask = np.array(inputs.known_covariate_mask, copy=True)
    missing_mask[-1, 2] = False
    missing_schedule = ForecastInputs(
        series_id=inputs.series_id,
        horizon=inputs.horizon,
        timestamps=inputs.timestamps,
        past_target=inputs.past_target,
        past_target_mask=inputs.past_target_mask,
        past_covariates=inputs.past_covariates,
        past_covariate_mask=inputs.past_covariate_mask,
        known_covariates=inputs.known_covariates,
        known_covariate_mask=missing_mask,
    )
    with pytest.raises(ValueError, match="fully observed"):
        dataset.counterfactual_target(missing_schedule)
