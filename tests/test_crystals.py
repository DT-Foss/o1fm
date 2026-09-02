from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module(name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


crystals = _load_module("o1fm_crystals_test", "src/o1fm/crystals.py")

ForecastCrystalBank = crystals.ForecastCrystalBank
ForecastCrystalConflictError = crystals.ForecastCrystalConflictError
ForecastCrystalKey = crystals.ForecastCrystalKey


def _key(regime: str = "stable") -> ForecastCrystalKey:
    return ForecastCrystalKey(
        model_identity="demo-model",
        feature_schema="lags+known-future/v1",
        graph_revision="graph-0001",
        regime=regime,
        horizon=2,
        target_count=1,
    )


def test_forecast_crystal_key_validates_profile_shape() -> None:
    with pytest.raises(ValueError):
        ForecastCrystalKey(
            model_identity="demo-model",
            feature_schema="lags+known-future/v1",
            graph_revision="graph-0001",
            regime="bad regime ",
            horizon=2,
            target_count=1,
        )


def test_crystal_bank_observe_predict_and_persist(tmp_path: Path) -> None:
    path = tmp_path / "crystals.json"
    bank = ForecastCrystalBank(
        path,
        ridge=1e-6,
        min_support=2,
        min_sequential_support=1,
        max_mean_absolute_error=10.0,
        min_coverage=0.0,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    key = _key()

    first = bank.observe(key, [0.0], [[1.0], [2.0]])
    assert first.ready is False
    assert first.support == 1
    assert first.sequential_support == 0

    second = bank.observe(key, [1.0], [[3.0], [4.0]])
    assert second.ready is True
    assert second.support == 2
    assert second.sequential_support == 1

    prediction = bank.predict(key, [2.0])
    assert prediction is not None
    np.testing.assert_allclose(
        prediction.point, np.array([[5.0], [6.0]], dtype=np.float32), atol=1e-3
    )
    assert list(prediction.quantiles) == [0.1, 0.5, 0.9]
    low = prediction.quantiles[0.1]
    mid = prediction.quantiles[0.5]
    high = prediction.quantiles[0.9]
    assert np.all(low <= mid)
    assert np.all(mid <= high)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema"] == "o1fm.forecast-crystal-bank/v2"
    assert persisted["entries"][0]["key"]["regime"] == "stable"

    reloaded = ForecastCrystalBank(
        path,
        ridge=1e-6,
        min_support=2,
        min_sequential_support=1,
        max_mean_absolute_error=10.0,
        min_coverage=0.0,
    )
    loaded_prediction = reloaded.predict(key, [2.0])
    assert loaded_prediction is not None
    np.testing.assert_allclose(loaded_prediction.point, prediction.point, atol=1e-6)


def test_crystal_bank_abstains_on_regime_mismatch_and_unready_profile(
    tmp_path: Path,
) -> None:
    bank = ForecastCrystalBank(
        tmp_path / "crystals.json", min_support=3, min_sequential_support=1
    )
    stable = _key("stable")
    bank.observe(stable, [0.0], [[1.0], [2.0]])
    bank.observe(stable, [1.0], [[3.0], [4.0]])

    assert bank.predict(stable, [2.0], regime="shifted") is None
    assert bank.predict(_key("shifted"), [2.0]) is None


def test_crystal_bank_rejects_non_finite_inputs(tmp_path: Path) -> None:
    bank = ForecastCrystalBank(tmp_path / "crystals.json")
    key = _key()
    with pytest.raises(ValueError):
        bank.observe(key, [np.inf], [[1.0], [2.0]])
    with pytest.raises(ValueError):
        bank.observe(key, [0.0], [[np.nan], [2.0]])
    bank.observe(
        key,
        [0.0],
        [[np.nan], [2.0]],
        future_target_mask=[[False], [True]],
    )


def test_crystal_bank_partial_mask_uses_zero_then_current_point_imputation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crystals.json"
    bank = ForecastCrystalBank(
        path,
        ridge=1e-6,
        min_support=2,
        min_sequential_support=1,
        max_mean_absolute_error=10.0,
        min_coverage=0.0,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    key = _key()

    first = bank.observe(
        key,
        [0.0],
        [[1.0], [999.0]],
        future_target_mask=[[True], [False]],
    )
    assert first.support == 1
    assert first.sequential_support == 0
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["entries"][0]["target_sum"] == [1.0, 0.0]

    second = bank.observe(key, [1.0], [[3.0], [4.0]])
    assert second.support == 2
    assert second.sequential_support == 1
    prediction = bank.predict(key, [2.0])
    assert prediction is not None
    np.testing.assert_allclose(
        prediction.point,
        np.array([[5.0], [8.0]], dtype=np.float32),
        atol=1e-3,
    )

    third = bank.observe(
        key,
        [2.0],
        [[5.0], [1234.0]],
        future_target_mask=[[True], [False]],
    )
    assert third.support == 3
    assert third.sequential_support == 2
    assert third.residual_support == 2
    assert third.mean_absolute_error == pytest.approx(2.0)
    assert third.empirical_coverage == pytest.approx(0.0)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    entry = persisted["entries"][0]
    assert entry["target_sum"][0] == pytest.approx(9.0)
    assert entry["target_sum"][1] == pytest.approx(12.0, abs=2e-5)
    assert entry["observed_values"] == 3
    assert entry["covered_values"] == 0
    assert entry["residual_rows"][-1][0] == pytest.approx(0.0, abs=1e-5)
    assert entry["residual_rows"][-1][1] == pytest.approx(0.0, abs=1e-9)

    reloaded = ForecastCrystalBank(
        path,
        ridge=1e-6,
        min_support=2,
        min_sequential_support=1,
        max_mean_absolute_error=10.0,
        min_coverage=0.0,
    )
    loaded_prediction = reloaded.predict(key, [2.0])
    assert loaded_prediction is not None
    assert np.all(loaded_prediction.quantiles[0.1] <= loaded_prediction.quantiles[0.5])
    assert np.all(loaded_prediction.quantiles[0.5] <= loaded_prediction.quantiles[0.9])


def test_crystal_bank_rejects_invalid_future_target_mask(tmp_path: Path) -> None:
    bank = ForecastCrystalBank(tmp_path / "crystals.json")
    key = _key()
    with pytest.raises(ValueError):
        bank.observe(
            key,
            [0.0],
            [[1.0], [2.0]],
            future_target_mask=[[True, False]],
        )


def test_crystal_bank_compiles_fit_target_and_scores_realized_residuals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "crystals.json"
    bank = ForecastCrystalBank(
        path,
        ridge=1e-6,
        min_support=2,
        min_sequential_support=1,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    key = _key()
    bank.observe(
        key,
        [0.0],
        [[100.0], [100.0]],
        fit_target=[[1.0], [2.0]],
    )
    stats = bank.observe(
        key,
        [1.0],
        [[103.0], [104.0]],
        fit_target=[[3.0], [4.0]],
    )

    prediction = bank.predict(key, [2.0])
    assert prediction is not None
    np.testing.assert_allclose(prediction.point, [[5.0], [6.0]], atol=1e-3)
    assert stats.mean_absolute_error == pytest.approx(102.0)

    persisted = json.loads(path.read_text(encoding="utf-8"))["entries"][0]
    assert persisted["target_sum"] == [4.0, 6.0]
    assert persisted["residual_rows"][-1] == pytest.approx([102.0, 102.0])

    before_support = stats.support
    scored = bank.observe(
        key,
        [2.0],
        [[7.0], [8.0]],
        fit_target=[[999.0], [999.0]],
        update_fit=False,
    )
    assert scored.support == before_support
    persisted = json.loads(path.read_text(encoding="utf-8"))["entries"][0]
    assert persisted["target_sum"] == [4.0, 6.0]

    with pytest.raises(ValueError, match="unfitted"):
        ForecastCrystalBank(tmp_path / "empty.json").observe(
            key,
            [0.0],
            [[1.0], [2.0]],
            update_fit=False,
        )

    coverage_bank = ForecastCrystalBank(
        tmp_path / "coverage.json",
        min_support=2,
        min_sequential_support=1,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    coverage_bank.observe(
        key,
        [0.0],
        [[1.0], [2.0]],
        fit_target=[[1.0], [2.0]],
    )
    coverage_stats = coverage_bank.observe(
        key,
        [1.0],
        [[1.0], [2.0]],
        fit_target=[[100.0], [200.0]],
    )
    assert coverage_stats.empirical_coverage == 1.0


def test_crystal_bank_compiles_residual_over_local_base(tmp_path: Path) -> None:
    bank = ForecastCrystalBank(
        tmp_path / "residual.json",
        ridge=1e-6,
        min_support=2,
        min_sequential_support=1,
        quantile_levels=(0.1, 0.5, 0.9),
    )
    key = _key()
    bank.observe(
        key,
        [0.0],
        [[11.0], [12.0]],
        fit_target=[[11.0], [12.0]],
        base_point=[[10.0], [10.0]],
    )
    bank.observe(
        key,
        [1.0],
        [[22.0], [24.0]],
        fit_target=[[22.0], [24.0]],
        base_point=[[20.0], [20.0]],
    )

    prediction = bank.predict(
        key,
        [2.0],
        base_point=[[30.0], [30.0]],
    )
    assert prediction is not None
    np.testing.assert_allclose(prediction.point, [[33.0], [36.0]], atol=1e-3)


def test_crystal_observation_ids_are_replay_safe_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    bank = ForecastCrystalBank(path, min_support=1, min_sequential_support=0)
    key = _key()
    first = bank.observe(
        key,
        [0.0],
        [[1.0], [2.0]],
        observation_id="ticket:1:crystal",
    )
    replay = bank.observe(
        key,
        [0.0],
        [[1.0], [2.0]],
        observation_id="ticket:1:crystal",
    )
    assert first.support == replay.support == 1

    restored = ForecastCrystalBank(path, min_support=1, min_sequential_support=0)
    persisted_replay = restored.observe(
        key,
        [0.0],
        [[1.0], [2.0]],
        observation_id="ticket:1:crystal",
    )
    assert persisted_replay.support == 1
    with pytest.raises(ForecastCrystalConflictError):
        restored.observe(
            key,
            [0.0],
            [[9.0], [9.0]],
            observation_id="ticket:1:crystal",
        )
