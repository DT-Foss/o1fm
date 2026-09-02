from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from o1fm.atlas import AtlasReceipt
from o1fm.branches import (
    BranchExecutionResult,
    InterventionPlan,
    InterventionSpec,
    JointEnvelope,
)
from o1fm.cartography import (
    ZERO_SHA256,
    CartographerBudgets,
    ForecastCartographer,
    ForecastProbeTarget,
)
from o1fm.cartography_runtime import (
    CartographyExecutionBridge,
    CartographyExecutionBudgetError,
    CartographyExecutionContractError,
    CartographyProbeNotReadyError,
    MeasuredBranchExecution,
    ProbeWindow,
    ShadowProbeLedger,
    execution_key,
)
from o1fm.contracts import ForecastInputs, ForecastPrediction, ForecastTarget


def _window(
    *,
    horizon: int = 4,
    error: float = 0.0,
    known_offset: float = 0.0,
    interventions: tuple[InterventionPlan, ...] = (),
    realized_intervention: str = ZERO_SHA256,
) -> ProbeWindow:
    history = 3
    timestamps = np.datetime64("2026-01-01T00") + np.arange(history + horizon).astype(
        "timedelta64[h]"
    )
    known = (np.arange(history + horizon, dtype=np.float32) + known_offset)[:, None]
    inputs = ForecastInputs(
        series_id="grid:north",
        horizon=horizon,
        timestamps=timestamps,
        past_target=np.zeros((history, 2), dtype=np.float32),
        past_covariates=np.ones((history, 1), dtype=np.float32),
        known_covariates=known,
    )
    base = known[history:, 0]
    values = np.stack((base + error, base + 10.0 + error), axis=1)
    target = ForecastTarget(
        series_id=inputs.series_id,
        timestamps=inputs.future_timestamps,
        values=values,
    )
    return ProbeWindow(
        inputs,
        target,
        interventions,
        realized_intervention_identity_sha256=realized_intervention,
    )


def _target(
    *,
    dataset: str = "1",
    horizon: int = 4,
    indices: tuple[int, ...] = (0, 1),
    intervention: str = ZERO_SHA256,
    seed: int = 7,
    expected_work: float = 1.0,
    expected_bytes: int = 8_000,
) -> ForecastProbeTarget:
    return ForecastProbeTarget(
        series_id="grid:north",
        dataset_identity_sha256=dataset * 64,
        model_identity_sha256="2" * 64,
        graph_topology_sha256="3" * 64,
        regime="stable",
        horizon=horizon,
        target_indices=indices,
        action_inventory=("foundation", "crystal"),
        intervention_identity_sha256=intervention,
        seed=seed,
        expected_work=expected_work,
        expected_bytes=expected_bytes,
    )


def _receipt() -> AtlasReceipt:
    return AtlasReceipt(
        segment_sha256="a" * 64,
        revision_sequence=12,
        revision_sha256="b" * 64,
        record_count=1,
        kind="forecast",
    )


def _forecast(
    calls: list[ForecastInputs],
    *,
    work: float = 2.0,
) -> Callable[[ForecastInputs], MeasuredBranchExecution]:
    def run(inputs: ForecastInputs) -> MeasuredBranchExecution:
        calls.append(inputs)
        known = inputs.known_covariates[inputs.history_length :, 0]
        point = np.stack((known, known + 10.0), axis=1)
        quantile_levels = (0.1, 0.5, 0.9)
        quantiles = point[..., None] + np.asarray((-1.0, 0.0, 1.0), dtype=np.float32)
        prediction = ForecastPrediction(
            series_id=inputs.series_id,
            timestamps=inputs.future_timestamps,
            point=point,
            quantiles=quantiles,
            quantile_levels=quantile_levels,
        )
        return MeasuredBranchExecution(
            BranchExecutionResult(
                prediction=prediction,
                action_identity="foundation",
                model_identity="toto-22m",
                profile_identity="profile:stable",
                route_identity="route:stable",
                graph_identity="graph:12",
                route_w1=0.25,
                atlas_citations=(_receipt(),),
                joint_envelope=JointEnvelope(
                    series_id=inputs.series_id,
                    timestamps=inputs.future_timestamps,
                    lower=point - 1.25,
                    upper=point + 1.25,
                ),
            ),
            work_cost=work,
        )

    return run


def _clock() -> Callable[[], float]:
    values = iter((100.0, 101.0, 102.0, 103.0, 104.0, 105.0))
    return lambda: next(values)


def test_execution_key_coalesces_only_physical_forward_invariants() -> None:
    short = _target(horizon=2, indices=(0,), expected_work=0.25)
    wide = _target(horizon=4, indices=(1,), expected_work=9.0, expected_bytes=99)
    assert execution_key(short) == execution_key(wide)

    assert execution_key(_target(seed=8)) != execution_key(wide)
    assert execution_key(_target(dataset="4")) != execution_key(wide)
    assert execution_key(_target(intervention="5" * 64)) != execution_key(wide)


def test_one_forward_satisfies_multiple_horizons_and_target_cells_exactly() -> None:
    window = _window()
    short = _target(horizon=2, indices=(0,), expected_work=1.0)
    wide = _target(horizon=4, indices=(1,), expected_work=1.0)
    calls: list[ForecastInputs] = []
    bridge = CartographyExecutionBridge(
        lambda _target: window,
        _forecast(calls),
        clock=_clock(),
    )
    bridge.register((wide, short))

    short_evidence = bridge(short)
    wide_evidence = bridge(wide)

    assert len(calls) == 1
    assert calls[0].horizon == 4
    assert short_evidence.observed_values == 2
    assert wide_evidence.observed_values == 4
    assert short_evidence.coverage_gap == pytest.approx(0.2)
    assert wide_evidence.coverage_gap == pytest.approx(0.2)
    assert short_evidence.joint_miss == 0.0
    assert wide_evidence.joint_miss == 0.0
    assert short_evidence.result_sha256 != wide_evidence.result_sha256
    assert short_evidence.atlas_sha256 == wide_evidence.atlas_sha256
    assert short_evidence.started_at == wide_evidence.started_at == 100.0
    assert short_evidence.completed_at == wide_evidence.completed_at == 101.0

    batch = bridge.batches[0]
    assert batch.forward_count == 1
    assert batch.physical_work == 2.0
    assert sum(batch.work_charges.values()) == 2.0
    assert sum(item.work_cost for item in batch.evidence.values()) == 2.0
    assert batch.payload_bytes > 0
    assert sum(batch.byte_charges.values()) == batch.payload_bytes
    assert bridge.metrics()["forward_count"] == 1


def test_intervention_branch_is_applied_before_the_single_forward() -> None:
    base = _window()
    spec = InterventionSpec(
        name="flat-price",
        operation="set",
        timestamps=base.inputs.future_timestamps,
        covariate_indices=(0,),
        values=np.full((base.inputs.horizon, 1), 5.0, dtype=np.float32),
    )
    plan = InterventionPlan("controlled", (spec,), observed_policy=True)
    window = ProbeWindow(
        base.inputs,
        base.target,
        (plan,),
        realized_intervention_identity_sha256=plan.identity,
    )
    target = _target(intervention=plan.identity, expected_work=2.0)
    calls: list[ForecastInputs] = []
    bridge = CartographyExecutionBridge(
        lambda _target: window,
        _forecast(calls),
        clock=_clock(),
    )

    evidence = bridge(target)

    assert evidence.target_sha256 == target.sha256
    assert len(calls) == 1
    np.testing.assert_array_equal(
        calls[0].known_covariates[calls[0].history_length :, 0],
        np.full(4, 5.0, dtype=np.float32),
    )

    factual_labels = ProbeWindow(base.inputs, base.target, (plan,))
    rejected = CartographyExecutionBridge(
        lambda _target: factual_labels,
        _forecast([]),
        clock=_clock(),
    )
    with pytest.raises(
        CartographyExecutionContractError,
        match="realized/oracle target labels",
    ):
        rejected(target)


def test_shadow_learning_crosses_windows_without_touching_live_authority() -> None:
    live_authority = {"settlements": 0}
    calls: list[ForecastInputs] = []
    first = _target(dataset="1", indices=(0,), expected_work=2.0)
    second = _target(dataset="4", indices=(0,), expected_work=2.0)
    windows = {
        first.sha256: _window(error=2.0),
        second.sha256: _window(error=0.5),
    }
    bridge = CartographyExecutionBridge(
        lambda target: windows[target.sha256],
        _forecast(calls),
        clock=_clock(),
    )

    first_evidence = bridge(first)
    second_evidence = bridge(second)

    assert first_evidence.o1_surprise == 0.0
    assert first_evidence.learning_progress == 0.0
    assert second_evidence.learning_progress > 0.0
    assert second_evidence.o1_surprise == 0.0
    assert len(bridge.shadow.snapshot()) == 1
    assert live_authority == {"settlements": 0}
    assert len(calls) == 2


def test_contract_and_work_capacity_fail_before_evidence_is_committed() -> None:
    left = _target(horizon=2, indices=(0,), expected_work=1.0)
    right = _target(horizon=4, indices=(1,), expected_work=1.0)
    left_window = _window()
    altered = _window(known_offset=100.0)
    calls: list[ForecastInputs] = []
    incompatible = CartographyExecutionBridge(
        lambda target: {
            left.sha256: left_window,
            right.sha256: altered,
        }[target.sha256],
        _forecast(calls),
        clock=_clock(),
    )
    with pytest.raises(
        CartographyExecutionContractError,
        match="exact model prefix",
    ):
        incompatible.execute_many((left, right))
    assert calls == []

    over_budget_calls: list[ForecastInputs] = []
    over_budget = CartographyExecutionBridge(
        lambda _target: left_window,
        _forecast(over_budget_calls, work=2.0),
        clock=_clock(),
    )
    with pytest.raises(CartographyExecutionBudgetError, match="expected_work"):
        over_budget(left)
    assert len(over_budget_calls) == 1
    assert over_budget.batches == ()
    assert over_budget.metrics()["cached_evidence"] == 0


def test_incompatible_registered_peer_cannot_poison_serial_execution() -> None:
    left = _target(horizon=2, indices=(0,), expected_work=2.0)
    right = _target(horizon=4, indices=(1,), expected_work=2.0)
    windows = {
        left.sha256: _window(),
        right.sha256: _window(known_offset=100.0),
    }
    calls: list[ForecastInputs] = []
    bridge = CartographyExecutionBridge(
        lambda target: windows[target.sha256],
        _forecast(calls),
        clock=_clock(),
    )
    bridge.register((left, right))

    assert bridge(left).target_sha256 == left.sha256
    assert len(calls) == 1
    assert bridge(right).target_sha256 == right.sha256
    assert len(calls) == 2
    assert bridge.metrics()["forward_count"] == 2


def test_unrealized_cell_is_rejected_before_compute() -> None:
    base = _window()
    target = _target(indices=(0,), expected_work=2.0)
    mask = np.ones_like(base.target.mask, dtype=np.bool_)
    mask[:, 0] = False
    pending = ProbeWindow(
        base.inputs,
        ForecastTarget(
            series_id=base.target.series_id,
            timestamps=base.target.timestamps,
            values=base.target.values,
            mask=mask,
        ),
    )
    calls: list[ForecastInputs] = []
    bridge = CartographyExecutionBridge(
        lambda _target: pending,
        _forecast(calls),
        clock=_clock(),
    )

    with pytest.raises(CartographyProbeNotReadyError, match="no realized"):
        bridge(target)
    assert calls == []
    assert bridge.shadow.snapshot() == {}


def test_scorer_failure_commits_neither_shadow_forward_nor_evidence() -> None:
    class FailingShadow(ShadowProbeLedger):
        def _transaction(self, entries, build):  # type: ignore[no-untyped-def]
            def fail_after_build(updates):  # type: ignore[no-untyped-def]
                build(updates)
                raise RuntimeError("synthetic scorer commit failure")

            return super()._transaction(entries, fail_after_build)

    target = _target(indices=(0,), expected_work=2.0)
    calls: list[ForecastInputs] = []
    shadow = FailingShadow()
    bridge = CartographyExecutionBridge(
        lambda _target: _window(),
        _forecast(calls),
        shadow=shadow,
        clock=_clock(),
    )

    with pytest.raises(RuntimeError, match="synthetic scorer"):
        bridge(target)
    assert len(calls) == 1
    assert shadow.snapshot() == {}
    assert bridge.batches == ()
    assert bridge.metrics()["execution_keys"] == 0
    assert bridge.metrics()["cached_evidence"] == 0


def test_bridge_is_directly_usable_by_serial_background_cartography(
    tmp_path: Path,
) -> None:
    window = _window()
    short = _target(horizon=2, indices=(0,), expected_work=1.0)
    wide = _target(horizon=4, indices=(1,), expected_work=1.0)
    calls: list[ForecastInputs] = []
    bridge = CartographyExecutionBridge(
        lambda _target: window,
        _forecast(calls),
        clock=_clock(),
        strict_byte_reservations=True,
    )
    bridge.register((short, wide))
    cartographer = ForecastCartographer(
        tmp_path / "map",
        budgets=CartographerBudgets(work=2.0, bytes=16_000),
    )
    cartographer.extend_targets((short, wide))

    handle = cartographer.background_run(bridge, max_jobs=2)
    receipts = handle.result(timeout=5.0)
    cartographer.close()

    assert len(receipts) == 2
    assert len(calls) == 1
    assert cartographer.metrics()["total_work"] == 2.0
    assert cartographer.metrics()["total_bytes"] == 16_000
    assert bridge.metrics()["forward_count"] == 1
    assert bridge.batches[0].ledger_bytes == 16_000
    assert bridge.batches[0].payload_bytes <= bridge.batches[0].ledger_bytes
    assert bridge.batches[0].reservation_shortfall_bytes == 0
    assert {receipt.target_sha256 for receipt in receipts} == {
        short.sha256,
        wide.sha256,
    }


def test_strict_byte_reservation_uses_exact_payload_size() -> None:
    target = _target(expected_work=2.0, expected_bytes=1)
    calls: list[ForecastInputs] = []
    bridge = CartographyExecutionBridge(
        lambda _target: _window(),
        _forecast(calls),
        clock=_clock(),
        strict_byte_reservations=True,
    )

    with pytest.raises(CartographyExecutionBudgetError, match="expected_bytes"):
        bridge(target)
    assert len(calls) == 1
    assert bridge.batches == ()
