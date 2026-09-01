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
    np.testing.assert_allclose(prediction.point, np.array([[5.0], [6.0]], dtype=np.float32), atol=1e-3)
    assert list(prediction.quantiles) == [0.1, 0.5, 0.9]
    low = prediction.quantiles[0.1]
    mid = prediction.quantiles[0.5]
    high = prediction.quantiles[0.9]
    assert np.all(low <= mid)
    assert np.all(mid <= high)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema"] == "o1fm.forecast-crystal-bank/v1"
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


def test_crystal_bank_abstains_on_regime_mismatch_and_unready_profile(tmp_path: Path) -> None:
    bank = ForecastCrystalBank(tmp_path / "crystals.json", min_support=3, min_sequential_support=1)
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
