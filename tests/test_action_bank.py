from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


action_bank = _load_module("o1fm_action_bank_test", "src/o1fm/action_bank.py")

ActionBank = action_bank.ActionBank
ActionProfileKey = action_bank.ActionProfileKey
ActionReceipt = action_bank.ActionReceipt


def _profile(regime: str = "stable") -> ActionProfileKey:
    return ActionProfileKey(
        model_identity="demo-model",
        feature_schema="lags+known-future/v1",
        graph_revision="graph-0001",
        regime=regime,
        horizon=2,
        target_count=1,
    )


def test_action_bank_recommends_cheapest_verified_action(tmp_path: Path) -> None:
    path = tmp_path / "action-bank.json"
    bank = ActionBank(
        path, quality_tolerance=0.05, coverage_tolerance=0.02, min_support=2
    )
    profile = _profile()

    bank.settle(
        ActionReceipt(
            profile,
            "foundation",
            quality=0.50,
            coverage=0.80,
            latency_seconds=2.0,
            forward_cost=4.0,
        )
    )
    bank.settle(
        ActionReceipt(
            profile,
            "foundation",
            quality=0.54,
            coverage=0.82,
            latency_seconds=2.2,
            forward_cost=4.1,
        )
    )
    bank.settle(
        ActionReceipt(
            profile,
            "crystal",
            quality=0.52,
            coverage=0.81,
            latency_seconds=0.1,
            forward_cost=0.0,
        )
    )
    bank.settle(
        ActionReceipt(
            profile,
            "crystal",
            quality=0.53,
            coverage=0.82,
            latency_seconds=0.1,
            forward_cost=0.0,
        )
    )

    recommendation = bank.recommend(profile)
    assert recommendation.action == "crystal"
    assert recommendation.reason == "verified-cheapest"
    assert recommendation.baseline is not None
    assert recommendation.chosen is not None
    assert (
        recommendation.chosen.mean_total_cost < recommendation.baseline.mean_total_cost
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema"] == "o1fm.action-bank/v1"

    reloaded = ActionBank(
        path, quality_tolerance=0.05, coverage_tolerance=0.02, min_support=2
    )
    assert reloaded.recommend(profile).action == "crystal"


def test_action_bank_falls_back_to_foundation_when_quality_is_worse(
    tmp_path: Path,
) -> None:
    bank = ActionBank(
        tmp_path / "action-bank.json", quality_tolerance=0.01, min_support=1
    )
    profile = _profile()
    bank.settle(
        ActionReceipt(
            profile,
            "foundation",
            quality=0.40,
            coverage=0.90,
            latency_seconds=1.0,
            forward_cost=3.0,
        )
    )
    bank.settle(
        ActionReceipt(
            profile,
            "crystal",
            quality=0.45,
            coverage=0.95,
            latency_seconds=0.1,
            forward_cost=0.0,
        )
    )

    recommendation = bank.recommend(profile)
    assert recommendation.action == "foundation"
    assert recommendation.reason in {"only-foundation-verified", "foundation-cheapest"}


def test_action_bank_unknown_profiles_return_foundation() -> None:
    bank = ActionBank()
    recommendation = bank.recommend(_profile())
    assert recommendation.action == "foundation"
    assert recommendation.reason == "unknown-profile"
    assert recommendation.baseline is None
    assert recommendation.chosen is None


def test_action_bank_validates_receipts() -> None:
    profile = _profile()
    with pytest.raises(ValueError):
        ActionReceipt(
            profile,
            "invalid",
            quality=0.1,
            coverage=0.5,
            latency_seconds=0.1,
            forward_cost=0.0,
        )


def test_action_bank_prices_verified_local_teacher(tmp_path: Path) -> None:
    bank = ActionBank(
        tmp_path / "actions.json",
        quality_tolerance=0.01,
        coverage_tolerance=0.02,
        min_support=1,
        latency_cost_weight=0.0,
        forward_cost_weight=1.0,
    )
    profile = _profile()
    bank.settle(ActionReceipt(profile, "foundation", 0.10, 0.80, 0.02, 1.0))
    bank.settle(ActionReceipt(profile, "local", 0.105, 0.81, 0.01, 0.2))

    recommendation = bank.recommend(profile)
    assert recommendation.action == "local"
    assert recommendation.chosen is not None
    assert recommendation.chosen.mean_forward_cost == 0.2
    with pytest.raises(ValueError):
        ActionReceipt(
            profile,
            "foundation",
            quality=0.1,
            coverage=1.5,
            latency_seconds=0.1,
            forward_cost=0.0,
        )
