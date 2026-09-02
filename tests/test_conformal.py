from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from o1fm.conformal import (
    ConformalApplication,
    ConformalConflictError,
    ConformalFieldConfig,
    ConformalFieldKey,
    ConformalIntegrityError,
    O1ConformalField,
)
from o1fm.contracts import ForecastPrediction, ForecastTarget


def _timestamps(horizon: int = 2) -> np.ndarray:
    return np.arange(
        np.datetime64("2026-01-01"),
        np.datetime64("2026-01-01") + horizon,
        dtype="datetime64[D]",
    )


def _prediction(
    *,
    horizon: int = 2,
    targets: int = 2,
    outer: tuple[float, float] = (0.0, 2.0),
) -> ForecastPrediction:
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    values = np.array([outer[0], 0.5, 1.0, 1.5, outer[1]], dtype=np.float32)
    quantiles = np.broadcast_to(values, (horizon, targets, len(levels))).copy()
    return ForecastPrediction(
        series_id="series-a",
        timestamps=_timestamps(horizon),
        point=np.ones((horizon, targets), dtype=np.float32),
        quantiles=quantiles,
        quantile_levels=levels,
    )


def _key(**changes: object) -> ConformalFieldKey:
    values: dict[str, object] = {
        "model_identity": "model:sha256:abc",
        "feature_schema": "features:v3",
        "graph_revision": "graph:001",
        "regime": "stable:0",
        "action": "foundation",
        "horizon": 2,
        "target_count": 2,
        "lower_quantile": 0.1,
        "upper_quantile": 0.9,
    }
    values.update(changes)
    return ConformalFieldKey(**values)


def _target(
    values: object,
    *,
    mask: object | None = None,
    horizon: int = 2,
) -> ForecastTarget:
    return ForecastTarget(
        series_id="series-a",
        timestamps=_timestamps(horizon),
        values=values,
        mask=mask,
    )


def test_cold_start_is_exactly_unchanged_and_arrays_are_immutable(
    tmp_path: Path,
) -> None:
    field = O1ConformalField(tmp_path)
    base = _prediction()

    application = field.apply(_key(), base)

    assert np.array_equal(application.prediction.point, base.point)
    assert np.array_equal(application.prediction.quantiles, base.quantiles)
    assert np.count_nonzero(application.lower_score_radius) == 0
    assert np.count_nonzero(application.upper_score_radius) == 0
    assert application.joint_score_radius == 0.0
    assert not application.scale.flags.writeable
    assert not application.joint_lower.flags.writeable
    with pytest.raises(ValueError):
        application.lower_expansion[0, 0] = 1.0
    assert field.metrics() == {
        "exact_profile_count": 0,
        "pooled_profile_count": 0,
        "settlements": 0,
        "observed_values": 0,
        "mean_marginal_coverage": 0.0,
        "joint_coverage": 0.0,
        "mean_marginal_width_expansion": 0.0,
    }


def test_directed_tails_expand_independently_and_keep_inner_quantiles(
    tmp_path: Path,
) -> None:
    field = O1ConformalField(tmp_path)
    key = _key()
    base = _prediction()
    first = field.apply(key, base)
    field.observe(
        key,
        first,
        _target([[-2.0, -1.0], [-3.0, -4.0]]),
        observation_id="lower-miss",
    )

    calibrated = field.apply(key, base)

    assert bool((calibrated.lower_score_radius > 0.0).all())
    assert np.count_nonzero(calibrated.upper_score_radius) == 0
    assert bool((calibrated.prediction.quantiles[:, :, 0] < 0.0).all())
    assert np.array_equal(
        calibrated.prediction.quantiles[:, :, 1:-1],
        base.quantiles[:, :, 1:-1],
    )
    assert np.array_equal(calibrated.prediction.point, base.point)
    assert bool((np.diff(calibrated.prediction.quantiles, axis=2) >= 0.0).all())


def test_pooled_evidence_transfers_to_new_action_graph_and_regime(
    tmp_path: Path,
) -> None:
    field = O1ConformalField(tmp_path)
    source = _key()
    base = _prediction()
    field.observe(
        source,
        field.apply(source, base),
        _target([[-2.0, -2.0], [-2.0, -2.0]]),
        observation_id="source",
    )
    destination = _key(
        action="crystal",
        graph_revision="graph:999",
        regime="shifted:4",
    )

    transferred = field.apply(destination, base)

    assert bool((transferred.lower_score_radius > 0.0).all())
    assert transferred.prior_settlements == 0
    assert transferred.lower_alpha == pytest.approx(
        field.apply(source, base).lower_alpha
    )
    assert field.metrics()["exact_profile_count"] == 1
    assert field.metrics()["pooled_profile_count"] == 1


def test_partial_mask_counts_observed_cells_and_persists_null_history(
    tmp_path: Path,
) -> None:
    field = O1ConformalField(tmp_path)
    key = _key()
    base = _prediction()
    mask = np.array([[True, False], [False, True]], dtype=np.bool_)
    update = field.observe(
        key,
        field.apply(key, base),
        _target([[-1.0, np.nan], [np.nan, 1.0]], mask=mask),
        observation_id="partial",
    )

    assert update.observed_values == 2
    assert update.marginal_hits == 1
    assert field.metrics()["observed_values"] == 2
    document = json.loads((tmp_path / "conformal-field.json").read_text())
    profile = next(iter(document["exact"].values()))
    assert profile["lower_history"][0][0] is not None
    assert profile["lower_history"][1] == [None]
    assert profile["lower_history"][2] == [None]
    assert profile["lower_history"][3][0] is not None


def test_aci_direction_and_o1_surprise_changes_learning_gain(tmp_path: Path) -> None:
    quiet = O1ConformalField(tmp_path / "quiet")
    surprised = O1ConformalField(tmp_path / "surprised")
    key = _key()
    base = _prediction()
    miss = _target([[-2.0, 4.0], [-2.0, 4.0]])
    quiet_update = quiet.observe(
        key,
        quiet.apply(key, base),
        miss,
        surprise=0.0,
        observation_id="miss",
    )
    surprised_update = surprised.observe(
        key,
        surprised.apply(key, base),
        miss,
        surprise=4.0,
        observation_id="miss",
    )

    assert quiet_update.lower_alpha < 0.1
    assert quiet_update.upper_alpha < 0.1
    assert quiet_update.joint_alpha < 0.2
    assert surprised_update.gain > quiet_update.gain
    assert surprised_update.lower_alpha < quiet_update.lower_alpha

    hit_application = quiet.apply(key, base)
    hit_update = quiet.observe(
        key,
        hit_application,
        _target([[1.0, 1.0], [1.0, 1.0]]),
        observation_id="hit",
    )
    assert hit_update.lower_alpha > quiet_update.lower_alpha
    assert hit_update.upper_alpha > quiet_update.upper_alpha
    assert hit_update.joint_alpha > quiet_update.joint_alpha


def test_joint_trajectory_envelope_is_prior_only_and_not_forced_into_marginal(
    tmp_path: Path,
) -> None:
    field = O1ConformalField(tmp_path)
    key = _key()
    base = _prediction()
    field.observe(
        key,
        field.apply(key, base),
        _target([[-4.0, 1.0], [1.0, 1.0]]),
        observation_id="one-path-miss",
    )

    application = field.apply(key, base)

    assert application.joint_score_radius > 0.0
    assert bool(
        (application.joint_lower <= application.prediction.quantiles[:, :, 0]).all()
    )
    assert bool(
        (application.joint_upper >= application.prediction.quantiles[:, :, -1]).all()
    )
    assert application.joint_lower[1, 1] < application.prediction.quantiles[1, 1, 0]
    assert not np.shares_memory(
        application.joint_lower, application.prediction.quantiles
    )


def test_persistence_reloads_profiles_metrics_and_calibration(tmp_path: Path) -> None:
    key = _key()
    base = _prediction()
    field = O1ConformalField(tmp_path)
    field.observe(
        key,
        field.apply(key, base),
        _target([[-1.0, -1.0], [-1.0, -1.0]]),
        observation_id="persisted",
    )
    expected = field.apply(key, base)
    expected_metrics = field.metrics()

    restored = O1ConformalField(tmp_path)
    actual = restored.apply(key, base)

    assert restored.metrics() == expected_metrics
    assert np.array_equal(actual.prediction.quantiles, expected.prediction.quantiles)
    assert actual.lower_alpha == expected.lower_alpha
    with pytest.raises(ConformalIntegrityError):
        O1ConformalField(tmp_path, config=ConformalFieldConfig(base_gain=0.03))


def test_observation_replay_is_idempotent_and_conflict_is_rejected(
    tmp_path: Path,
) -> None:
    field = O1ConformalField(tmp_path)
    key = _key()
    application = field.apply(key, _prediction())
    target = _target([[-1.0, 1.0], [1.0, 1.0]])
    first = field.observe(key, application, target, observation_id="obs:42")
    before = field.metrics()

    replay = field.observe(key, application, target, observation_id="obs:42")

    assert replay.replayed
    assert replay.content_sha256 == first.content_sha256
    assert field.metrics() == before
    with pytest.raises(ConformalConflictError):
        field.observe(
            key,
            application,
            _target([[-2.0, 1.0], [1.0, 1.0]]),
            observation_id="obs:42",
        )


@pytest.mark.parametrize(
    "options",
    [
        {"target_coverage": 1.0},
        {"base_gain": 0.0},
        {"surprise_gain": -1.0},
        {"max_history": 0},
        {"min_scale": 0.0},
        {"alpha_min": 0.5, "alpha_max": 0.5},
    ],
)
def test_invalid_config_is_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ConformalFieldConfig(**options)


def test_invalid_keys_predictions_targets_and_nonfinite_o1_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        _key(action="unknown")
    with pytest.raises(ValueError):
        _key(lower_quantile=0.9, upper_quantile=0.1)
    field = O1ConformalField(tmp_path)
    base = _prediction()
    with pytest.raises(ValueError):
        field.apply(_key(horizon=1), base)
    wrong_levels = ForecastPrediction(
        series_id=base.series_id,
        timestamps=base.timestamps,
        point=base.point,
        quantiles=base.quantiles[:, :, 1:],
        quantile_levels=base.quantile_levels[1:],
    )
    with pytest.raises(ValueError):
        field.apply(_key(), wrong_levels)
    application = field.apply(_key(), base)
    with pytest.raises(ValueError):
        field.observe(
            _key(),
            application,
            _target([[1.0, 1.0]] * 2),
            surprise=float("nan"),
        )
    with pytest.raises(ValueError):
        field.observe(
            _key(),
            application,
            _target([[1.0, 1.0]] * 2),
            cusum=float("inf"),
        )
    with pytest.raises(ValueError):
        field.observe(
            _key(),
            application,
            _target([[1.0]], horizon=1),
        )


def test_application_rejects_nonfinite_or_wrong_shape_arrays(tmp_path: Path) -> None:
    key = _key()
    base = _prediction()
    zeros = np.zeros(key.shape, dtype=np.float32)
    with pytest.raises(ValueError):
        ConformalApplication(
            key=key,
            base_prediction=base,
            prediction=base,
            scale=np.full(key.shape, np.nan),
            lower_score_radius=zeros,
            upper_score_radius=zeros,
            lower_expansion=zeros,
            upper_expansion=zeros,
            joint_score_radius=0.0,
            joint_lower=zeros,
            joint_upper=zeros,
            lower_alpha=0.1,
            upper_alpha=0.1,
            joint_alpha=0.2,
            prior_settlements=0,
        )
    with pytest.raises(ValueError):
        replace(
            O1ConformalField(tmp_path).apply(key, base),
            scale=np.zeros((1, 1), dtype=np.float32),
        )
