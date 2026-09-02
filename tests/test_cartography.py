from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from o1fm.cartography import (
    ZERO_SHA256,
    CartographerBudgets,
    CartographerConfig,
    CartographyBudgetError,
    CartographyIntegrityError,
    ForecastCartographer,
    ForecastProbeTarget,
    ProbeConflictError,
    ProbeEvidence,
)


def _target(
    seed: int,
    *,
    series: str = "load:north",
    topology: str = "3" * 64,
    horizon: int = 12,
    target_indices: tuple[int, ...] = (0,),
    actions: tuple[str, ...] = ("foundation", "crystal"),
    intervention: str = ZERO_SHA256,
    expected_work: float = 1.0,
    expected_bytes: int = 10,
) -> ForecastProbeTarget:
    return ForecastProbeTarget(
        series_id=series,
        dataset_identity_sha256="1" * 64,
        model_identity_sha256="2" * 64,
        graph_topology_sha256=topology,
        regime="stable",
        horizon=horizon,
        target_indices=target_indices,
        action_inventory=actions,
        intervention_identity_sha256=intervention,
        seed=seed,
        expected_work=expected_work,
        expected_bytes=expected_bytes,
    )


def _evidence(
    target: ForecastProbeTarget,
    *,
    coverage_gap: float = 0.1,
    joint_miss: float = 0.0,
    surprise: float = 0.0,
    learning: float = 0.0,
    route_w1: float = 0.0,
    work: float = 0.5,
    salt: int = 0,
) -> ProbeEvidence:
    return ProbeEvidence(
        target_sha256=target.sha256,
        coverage_gap=coverage_gap,
        pinball=0.2,
        mae=0.3,
        joint_miss=joint_miss,
        o1_surprise=surprise,
        learning_progress=learning,
        route_w1=route_w1,
        observed_values=4,
        work_cost=work,
        result_sha256=f"{4 + salt:x}" * 64,
        atlas_sha256=f"{8 + salt:x}" * 64,
        started_at=100.0 + salt,
        completed_at=101.0 + salt,
    )


def test_probe_target_and_evidence_have_exact_canonical_identities() -> None:
    ordered = _target(7, target_indices=(2, 0, 2), actions=("crystal", "foundation"))
    canonical = _target(7, target_indices=(0, 2), actions=("foundation", "crystal"))

    assert ordered == canonical
    assert ordered.sha256 == canonical.identity_sha256
    assert len(ordered.sha256) == 64
    assert _target(8).sha256 != _target(7).sha256
    evidence = _evidence(ordered)
    assert ProbeEvidence.from_record(evidence.to_record()) == evidence
    assert evidence.sha256 == evidence.identity_sha256


def test_idempotent_extension_and_productive_priority_ordering(tmp_path: Path) -> None:
    config = CartographerConfig(unexplored_weight=1.0)
    cartographer = ForecastCartographer(tmp_path / "map", config=config)
    high_history = _target(0, series="high")
    low_history = _target(0, series="low")
    high_next = _target(1, series="high")
    low_next = _target(1, series="low")

    assert cartographer.extend_targets([high_history, low_history]) == 2
    assert cartographer.extend_targets([high_history, low_history]) == 0
    cartographer.settle(
        _evidence(
            high_history,
            coverage_gap=0.8,
            joint_miss=1.0,
            surprise=2.0,
            route_w1=1.0,
        ),
        committed_at=200.0,
    )
    cartographer.settle(_evidence(low_history), committed_at=201.0)
    cartographer.extend_targets([low_next, high_next])

    high_priority = cartographer.priority_breakdown(high_next)
    low_priority = cartographer.priority_breakdown(low_next)
    assert high_priority.priority > low_priority.priority
    assert high_priority.related_receipts == 1
    assert low_priority.low_learning_decay == pytest.approx(0.5)
    assert cartographer.next_job() == high_next


def test_priority_accounts_for_unexplored_cells_and_expected_cost(
    tmp_path: Path,
) -> None:
    cartographer = ForecastCartographer(tmp_path / "map")
    history = _target(0)
    repeated = _target(1)
    novel = _target(2, horizon=24, target_indices=(1,), actions=("intervene",))
    expensive = _target(3, horizon=48, expected_work=100.0)
    cartographer.extend_targets([history])
    cartographer.settle(_evidence(history), committed_at=200.0)
    cartographer.extend_targets([repeated, novel, expensive])

    assert cartographer.priority_breakdown(novel).unexplored_fraction == 0.75
    assert cartographer.priority(novel) > cartographer.priority(repeated)
    assert cartographer.value_density(expensive) < cartographer.value_density(novel)


def test_priority_evidence_isolated_across_graph_topologies(tmp_path: Path) -> None:
    config = CartographerConfig(unexplored_weight=1.0)
    cartographer = ForecastCartographer(tmp_path / "map", config=config)
    old_topology = _target(0, topology="3" * 64)
    same_topology = _target(1, topology="3" * 64)
    new_topology = _target(2, topology="4" * 64)
    cartographer.extend_targets((old_topology, same_topology, new_topology))
    cartographer.settle(
        _evidence(
            old_topology,
            coverage_gap=0.9,
            joint_miss=1.0,
            surprise=4.0,
            learning=0.0,
            route_w1=3.0,
        ),
        committed_at=200.0,
    )

    related = cartographer.priority_breakdown(same_topology)
    isolated = cartographer.priority_breakdown(new_topology)
    fresh = ForecastCartographer(tmp_path / "fresh", config=config)
    fresh.add_target(new_topology)
    baseline = fresh.priority_breakdown(new_topology)

    assert related.related_receipts == 1
    assert related.low_learning_decay == pytest.approx(0.5)
    assert isolated.related_receipts == 0
    assert isolated.unexplored_fraction == 1.0
    assert isolated.low_learning_decay == 1.0
    assert isolated.priority == pytest.approx(baseline.priority)
    assert isolated.value_density == pytest.approx(baseline.value_density)


def test_d_optimal_batch_prefers_information_diversity(tmp_path: Path) -> None:
    cartographer = ForecastCartographer(tmp_path / "map")
    axis_x = _target(1)
    duplicate_x = _target(2)
    axis_y = _target(3)
    cartographer.extend_targets([axis_x, duplicate_x, axis_y])

    selected = cartographer.d_optimal_batch(
        {
            axis_x: np.array([10.0, 0.0]),
            duplicate_x: np.array([9.0, 0.0]),
            axis_y: np.array([0.0, 8.0]),
        },
        batch_size=2,
    )

    assert selected == (axis_x, axis_y)
    assert cartographer.acquire_batch(
        {target: [1.0, 0.0] for target in selected}, batch_size=1
    )


def test_global_and_batch_budgets_are_enforced_exactly(tmp_path: Path) -> None:
    cartographer = ForecastCartographer(
        tmp_path / "map",
        budgets=CartographerBudgets(work=2.0, bytes=20),
    )
    first = _target(1, expected_work=1.5, expected_bytes=10)
    second = _target(2, expected_work=1.5, expected_bytes=10)
    cartographer.extend_targets([first, second])
    cartographer.settle(_evidence(first, work=1.5), committed_at=200.0)

    assert cartographer.next_job() is None
    assert cartographer.metrics()["remaining_work"] == pytest.approx(0.5)
    assert cartographer.d_optimal_batch({second: [1.0]}, batch_size=1) == ()
    with pytest.raises(CartographyBudgetError):
        cartographer.settle(_evidence(second, work=1.5), committed_at=201.0)

    byte_limited = ForecastCartographer(
        tmp_path / "bytes",
        work_budget=10.0,
        byte_budget=10,
    )
    too_large = _target(3, expected_bytes=11)
    byte_limited.add_target(too_large)
    assert byte_limited.next_job() is None


def test_evidence_replay_is_idempotent_and_conflict_detected(tmp_path: Path) -> None:
    cartographer = ForecastCartographer(tmp_path / "map")
    target = _target(1)
    evidence = _evidence(target)
    cartographer.add_target(target)

    first = cartographer.settle(evidence, committed_at=200.0)
    replay = cartographer.settle(evidence, committed_at=999.0)
    assert replay == first
    assert cartographer.metrics()["sequence"] == 1
    with pytest.raises(ProbeConflictError):
        cartographer.settle(
            _evidence(target, coverage_gap=0.2),
            committed_at=201.0,
        )


def test_checkpoint_reload_and_authenticated_history(tmp_path: Path) -> None:
    root = tmp_path / "map"
    target = _target(1)
    cartographer = ForecastCartographer(root, capacity=5)
    cartographer.add_target(target)
    receipt = cartographer.settle(_evidence(target), committed_at=200.0)

    restored = ForecastCartographer(root)
    assert restored.receipts == (receipt,)
    assert restored.targets == (target,)
    assert restored.metrics() == cartographer.metrics()
    restored.reload()

    event = next((root / "evidence").glob("*.json"))
    document = json.loads(event.read_text(encoding="utf-8"))
    document["receipt"]["evidence"]["mae"] = (9.0).hex()
    event.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(CartographyIntegrityError):
        ForecastCartographer(root)


def test_step_and_bounded_run_execute_external_callback(tmp_path: Path) -> None:
    cartographer = ForecastCartographer(tmp_path / "map")
    targets = [_target(index) for index in range(3)]
    cartographer.extend_targets(targets)
    calls: list[str] = []

    def executor(target: ForecastProbeTarget) -> ProbeEvidence:
        calls.append(target.sha256)
        return _evidence(target, salt=len(calls))

    first = cartographer.step(executor)
    rest = cartographer.run(executor, max_jobs=1)
    assert first is not None
    assert len(rest) == 1
    assert len(calls) == 2
    assert cartographer.metrics()["pending"] == 1
    assert len(cartographer.run(executor, max_jobs=10, maximum_work=0.0)) == 0


def test_background_runner_is_nonblocking_and_strictly_serial(tmp_path: Path) -> None:
    cartographer = ForecastCartographer(tmp_path / "map")
    cartographer.extend_targets([_target(index) for index in range(4)])
    guard = threading.Lock()
    active = 0
    maximum_active = 0

    def executor(target: ForecastProbeTarget) -> ProbeEvidence:
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return _evidence(target, salt=target.seed)

    first = cartographer.submit_background(executor, max_jobs=2)
    second = cartographer.submit_background(executor, max_jobs=2)
    assert not second.done()
    assert len(first.result(timeout=3.0)) == 2
    assert len(second.result(timeout=3.0)) == 2
    assert maximum_active == 1
    assert cartographer.metrics()["completed"] == 4
    cartographer.close()


def test_metrics_are_exact_over_committed_evidence(tmp_path: Path) -> None:
    config = CartographerConfig(
        coverage_weight=2.0,
        joint_miss_weight=3.0,
        surprise_weight=4.0,
        route_w1_weight=5.0,
        learning_progress_weight=6.0,
    )
    cartographer = ForecastCartographer(tmp_path / "map", config=config)
    target = _target(1, expected_bytes=12)
    cartographer.add_target(target)
    cartographer.settle(
        _evidence(
            target,
            coverage_gap=-0.5,
            joint_miss=1.0,
            surprise=2.0,
            route_w1=3.0,
            learning=-0.25,
            work=0.75,
        ),
        committed_at=200.0,
    )

    metrics = cartographer.metrics()
    assert metrics["frontier"] == 0
    assert metrics["pending"] == 0
    assert metrics["completed"] == 1
    assert metrics["total_work"] == pytest.approx(0.75)
    assert metrics["total_bytes"] == 12
    assert metrics["coverage_debt"] == pytest.approx(0.5)
    assert metrics["coverage_debt_sum"] == pytest.approx(2.0)
    assert metrics["information_gain"] == pytest.approx(
        2 * 0.5 + 3 * 1.0 + 4 * 2.0 + 5 * 3.0 + 6 * 0.25
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("coverage_gap", np.nan),
        ("pinball", np.inf),
        ("mae", -1.0),
        ("joint_miss", 2.0),
        ("o1_surprise", -1.0),
        ("learning_progress", np.nan),
        ("route_w1", np.inf),
        ("work_cost", -1.0),
    ],
)
def test_evidence_rejects_nonfinite_or_invalid_metrics(
    field: str,
    value: float,
) -> None:
    target = _target(1)
    arguments = {
        "target_sha256": target.sha256,
        "coverage_gap": 0.1,
        "pinball": 0.2,
        "mae": 0.3,
        "joint_miss": 0.0,
        "o1_surprise": 0.0,
        "learning_progress": 0.0,
        "route_w1": 0.0,
        "observed_values": 1,
        "work_cost": 0.5,
        "result_sha256": "4" * 64,
        "atlas_sha256": "5" * 64,
        "started_at": 1.0,
        "completed_at": 2.0,
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        ProbeEvidence(**arguments)


def test_evidence_rejects_reversed_timestamps() -> None:
    target = _target(1)
    with pytest.raises(ValueError, match="completed_at"):
        ProbeEvidence(
            target_sha256=target.sha256,
            coverage_gap=0.0,
            pinball=0.0,
            mae=0.0,
            joint_miss=0.0,
            o1_surprise=0.0,
            learning_progress=0.0,
            route_w1=0.0,
            observed_values=1,
            work_cost=0.0,
            result_sha256="4" * 64,
            atlas_sha256="5" * 64,
            started_at=2.0,
            completed_at=1.0,
        )


def test_invalid_features_and_executor_outputs_do_not_settle(tmp_path: Path) -> None:
    cartographer = ForecastCartographer(tmp_path / "map")
    target = _target(1)
    cartographer.add_target(target)
    with pytest.raises(ValueError, match="finite"):
        cartographer.d_optimal_batch({target: [np.nan]}, batch_size=1)
    with pytest.raises(ValueError, match="share one dimension"):
        second = _target(2)
        cartographer.add_target(second)
        cartographer.d_optimal_batch({target: [1.0], second: [1.0, 2.0]}, batch_size=1)
    with pytest.raises(TypeError, match="ProbeEvidence"):
        cartographer.step(lambda _: object())  # type: ignore[return-value]
    assert cartographer.metrics()["completed"] == 0
