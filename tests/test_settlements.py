from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from o1fm.contracts import ForecastTarget
from o1fm.settlements import SettlementConflictError, SettlementLedger


def _target(mask: object, values: object | None = None) -> ForecastTarget:
    selected_values = (
        np.asarray(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)), dtype=np.float32)
        if values is None
        else values
    )
    return ForecastTarget(
        series_id="series-a",
        timestamps=np.datetime64("2026-01-01T00")
        + np.arange(3).astype("timedelta64[h]"),
        values=selected_values,
        mask=mask,
    )


def test_ledger_emits_only_new_cells_from_growing_targets(tmp_path: Path) -> None:
    ledger = SettlementLedger(tmp_path / "settlements.json")
    first_target = _target(((True, False), (False, False), (False, False)))
    growing_target = _target(((True, False), (True, True), (False, False)))

    with ledger.transaction():
        first = ledger.prepare("ticket-a", first_target)
        assert first.observed_values == 1
        assert not first.replayed
        ledger.commit(first)

    with ledger.transaction():
        second = ledger.prepare("ticket-a", growing_target)
        assert second.observed_values == 2
        assert not second.replayed
        np.testing.assert_array_equal(
            second.target.mask,
            ((False, False), (True, True), (False, False)),
        )
        ledger.commit(second)

    restored = SettlementLedger(tmp_path / "settlements.json")
    with restored.transaction():
        replay = restored.prepare("ticket-a", growing_target)
        assert replay.replayed
        assert replay.fragment_sha256 == second.fragment_sha256
        assert replay.observed_values == 2
        restored.commit(replay)


def test_prepare_is_crash_replayable_before_ledger_commit(tmp_path: Path) -> None:
    path = tmp_path / "settlements.json"
    target = _target(((True, True), (False, False), (False, False)))
    ledger = SettlementLedger(path)
    with ledger.transaction():
        prepared = ledger.prepare("ticket-a", target)

    restored = SettlementLedger(path)
    with restored.transaction():
        repeated = restored.prepare("ticket-a", target)
        assert repeated.fragment_sha256 == prepared.fragment_sha256
        assert not repeated.replayed
        restored.commit(repeated)


def test_ledger_rejects_changed_observed_values_and_contracts(tmp_path: Path) -> None:
    ledger = SettlementLedger(tmp_path / "settlements.json")
    initial = _target(((True, False), (False, False), (False, False)))
    with ledger.transaction():
        fragment = ledger.prepare("ticket-a", initial)
        ledger.commit(fragment)

    changed = np.asarray(((9.0, 2.0), (3.0, 4.0), (5.0, 6.0)), dtype=np.float32)
    with ledger.transaction(), pytest.raises(SettlementConflictError):
        ledger.prepare(
            "ticket-a",
            _target(((True, False), (False, False), (False, False)), changed),
        )

    other_series = ForecastTarget(
        series_id="series-b",
        timestamps=initial.timestamps,
        values=initial.values,
        mask=initial.mask,
    )
    with ledger.transaction(), pytest.raises(SettlementConflictError):
        ledger.prepare("ticket-a", other_series)


def test_new_cumulative_request_without_new_cells_replays_last_fragment(
    tmp_path: Path,
) -> None:
    ledger = SettlementLedger(tmp_path / "settlements.json")
    left = _target(((True, False), (False, False), (False, False)))
    right = _target(((False, True), (False, False), (False, False)))
    union = _target(((True, True), (False, False), (False, False)))
    with ledger.transaction():
        first = ledger.prepare("ticket-a", left)
        ledger.commit(first)
    with ledger.transaction():
        second = ledger.prepare("ticket-a", right)
        ledger.commit(second)
    with ledger.transaction():
        replay = ledger.prepare("ticket-a", union)
        assert replay.replayed
        assert replay.fragment_sha256 == second.fragment_sha256
        ledger.commit(replay)
