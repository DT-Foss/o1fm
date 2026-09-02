from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from o1fm.contracts import ForecastPrediction, ForecastTarget
from o1fm.intervention_crystals import (
    InterventionResponseConflictError,
    InterventionResponseCrystal,
    InterventionResponseKey,
)


def _key() -> InterventionResponseKey:
    return InterventionResponseKey(
        model_identity="1" * 64,
        feature_schema="2" * 64,
        intervention_schema="4" * 64,
        target_schema="5" * 64,
        graph_revision="3" * 64,
        regime="regime:0",
        horizon=2,
        target_count=1,
    )


def _timestamps() -> np.ndarray:
    return np.datetime64("2026-01-01T00") + np.arange(2).astype("timedelta64[h]")


def _factual(point: float = 10.0) -> ForecastPrediction:
    values = np.full((2, 1), point, dtype=np.float32)
    offsets = np.asarray((-0.2, 0.0, 0.2), dtype=np.float32)
    return ForecastPrediction(
        series_id="series-a",
        timestamps=_timestamps(),
        point=values,
        quantiles=values[..., None] + offsets,
        quantile_levels=(0.1, 0.5, 0.9),
    )


def _target(
    effect: float, *, mask: object | None = None
) -> tuple[ForecastTarget, ForecastTarget]:
    factual = ForecastTarget(
        series_id="series-a",
        timestamps=_timestamps(),
        values=np.full((2, 1), 10.0, dtype=np.float32),
        mask=mask,
    )
    intervention = ForecastTarget(
        series_id="series-a",
        timestamps=_timestamps(),
        values=np.asarray(((10.0 + effect,), (10.0 + 2.0 * effect,)), dtype=np.float32),
        mask=mask,
    )
    return factual, intervention


def _bank(path: Path) -> InterventionResponseCrystal:
    return InterventionResponseCrystal(
        path,
        ridge=1e-4,
        min_support=3,
        min_sequential_support=2,
        max_mean_effect_error=1.0,
        min_direction_cosine=0.5,
        min_coverage=0.5,
        quantile_levels=(0.1, 0.5, 0.9),
    )


def test_response_crystal_learns_effect_and_corrects_factual_teacher(
    tmp_path: Path,
) -> None:
    bank = _bank(tmp_path / "response")
    key = _key()
    factual_prediction = _factual()
    for index, value in enumerate((-2.0, 2.0, -1.0, 1.0, 0.0, 0.5, -0.5)):
        factual, intervention = _target(2.0 * value)
        bank.observe(
            key,
            (1.0, value),
            factual_prediction,
            factual,
            intervention,
            observation_id=f"obs:{index}",
        )

    prediction = bank.predict(key, (1.0, 1.5), factual_prediction)
    assert prediction is not None
    np.testing.assert_allclose(prediction.effect_point[:, 0], (3.0, 6.0), atol=2e-3)
    np.testing.assert_allclose(
        prediction.corrected.point[:, 0],
        (13.0, 16.0),
        atol=2e-3,
    )
    assert prediction.stats.ready
    assert prediction.stats.effect_direction_cosine is not None
    assert prediction.stats.effect_direction_cosine > 0.99
    assert prediction.stats.mean_effect_error is not None
    assert prediction.stats.mean_effect_error < 1.0


def test_partial_effects_replay_persist_and_conflict_exactly_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "response"
    bank = _bank(root)
    key = _key()
    mask = np.asarray(((True,), (False,)), dtype=np.bool_)
    factual, intervention = _target(1.0, mask=mask)
    first = bank.observe(
        key,
        (1.0, 1.0),
        _factual(),
        factual,
        intervention,
        observation_id="partial:1",
    )
    replay = bank.observe(
        key,
        (1.0, 1.0),
        _factual(),
        factual,
        intervention,
        observation_id="partial:1",
    )
    assert first.support == replay.support == 1

    restored = _bank(root)
    assert restored.describe(key).support == 1
    persisted = restored.observe(
        key,
        (1.0, 1.0),
        _factual(),
        factual,
        intervention,
        observation_id="partial:1",
    )
    assert persisted.support == 1
    changed_factual, changed = _target(3.0, mask=mask)
    with pytest.raises(InterventionResponseConflictError):
        restored.observe(
            key,
            (1.0, 1.0),
            _factual(),
            changed_factual,
            changed,
            observation_id="partial:1",
        )


def test_prepared_observation_recovers_after_effect_bank_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "response"
    bank = _bank(root)
    key = _key()
    factual, intervention = _target(1.0)
    original = bank.effects._persist

    def fail_once() -> None:
        raise RuntimeError("injected failure")

    monkeypatch.setattr(bank.effects, "_persist", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        bank.observe(
            key,
            (1.0, 1.0),
            _factual(),
            factual,
            intervention,
            observation_id="recover:1",
        )
    monkeypatch.setattr(bank.effects, "_persist", original)
    recovered = bank.observe(
        key,
        (1.0, 1.0),
        _factual(),
        factual,
        intervention,
        observation_id="recover:1",
    )
    assert recovered.support == 1
    assert _bank(root).describe(key).support == 1
