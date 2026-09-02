"""Persistent exactly-once fragments for incrementally realized forecasts."""

from __future__ import annotations

import fcntl
import json
import os
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .contracts import ForecastTarget
from .identity import canonical_json_bytes, require_sha256, sha256_json

SETTLEMENT_LEDGER_SCHEMA = "o1fm.settlement-ledger/v1"
SETTLEMENT_FRAGMENT_SCHEMA = "o1fm.settlement-fragment/v1"
SETTLEMENT_REQUEST_SCHEMA = "o1fm.settlement-request/v1"
MAX_STATE_BYTES = 128 * 1024 * 1024
MAX_TICKETS = 1_000_000
MAX_FRAGMENTS_PER_TICKET = 1_000_000

_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


class SettlementLedgerError(RuntimeError):
    """Persistent settlement state cannot satisfy the requested transition."""


class SettlementConflictError(SettlementLedgerError):
    """Previously observed target evidence was supplied with another value."""


class SettlementIntegrityError(SettlementLedgerError):
    """The settlement ledger is malformed or not canonically encoded."""


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _ticket_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 4096
    ):
        raise ValueError("ticket_id must be canonical bounded text")
    return value


def _timestamps_ns(target: ForecastTarget) -> list[int]:
    return [
        int(value)
        for value in target.timestamps.astype("datetime64[ns]").astype(np.int64)
    ]


def _target_record(target: ForecastTarget) -> dict[str, object]:
    return {
        "mask": target.mask.tolist(),
        "series_id": target.series_id,
        "timestamps_ns": _timestamps_ns(target),
        "values": target.values.tolist(),
    }


def _request_sha256(ticket_id: str, target: ForecastTarget) -> str:
    return sha256_json(
        {
            "schema": SETTLEMENT_REQUEST_SCHEMA,
            "target": _target_record(target),
            "ticket_id": ticket_id,
        }
    )


def _fragment_sha256(ticket_id: str, target: ForecastTarget) -> str:
    return sha256_json(
        {
            "schema": SETTLEMENT_FRAGMENT_SCHEMA,
            "target": _target_record(target),
            "ticket_id": ticket_id,
        }
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SettlementFragment:
    """The previously unseen cells carried by one settlement request."""

    ticket_id: str
    request_sha256: str
    fragment_sha256: str
    target: ForecastTarget
    replayed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticket_id", _ticket_id(self.ticket_id))
        object.__setattr__(
            self,
            "request_sha256",
            require_sha256(self.request_sha256, field="request_sha256"),
        )
        object.__setattr__(
            self,
            "fragment_sha256",
            require_sha256(self.fragment_sha256, field="fragment_sha256"),
        )
        if not isinstance(self.target, ForecastTarget):
            raise TypeError("target must be ForecastTarget")
        if not isinstance(self.replayed, bool):
            raise TypeError("replayed must be boolean")
        if not bool(self.target.mask.any()):
            raise ValueError("a settlement fragment must observe at least one value")
        if _fragment_sha256(self.ticket_id, self.target) != self.fragment_sha256:
            raise ValueError("fragment identity differs from its target evidence")

    @property
    def observed_values(self) -> int:
        return int(self.target.mask.sum())


class SettlementLedger:
    """Crash-replayable union of target cells already settled per ticket.

    A transaction lock is intentionally held while runtime components consume a
    prepared fragment.  If the process stops before ``commit()``, the next run
    prepares the same fragment identity and every downstream component replays
    its already-written receipt instead of counting the values twice.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().absolute()
        if not self.path.name:
            raise ValueError("settlement ledger path must name a file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._thread_lock = _lock_for(self.path)
        self._document: dict[str, object] | None = None

    @staticmethod
    def _empty() -> dict[str, object]:
        return {"schema": SETTLEMENT_LEDGER_SCHEMA, "tickets": []}

    def _load(self) -> dict[str, object]:
        if not self.path.exists():
            return self._empty()
        try:
            payload = self.path.read_bytes()
        except OSError as exc:
            raise SettlementIntegrityError("cannot read settlement ledger") from exc
        if not payload or len(payload) > MAX_STATE_BYTES:
            raise SettlementIntegrityError("settlement ledger size is invalid")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SettlementIntegrityError("settlement ledger is not JSON") from exc
        if (
            not isinstance(document, dict)
            or set(document) != {"schema", "tickets"}
            or document.get("schema") != SETTLEMENT_LEDGER_SCHEMA
            or not isinstance(document.get("tickets"), list)
            or len(document["tickets"]) > MAX_TICKETS
        ):
            raise SettlementIntegrityError("settlement ledger envelope is invalid")
        try:
            if canonical_json_bytes(document) != payload:
                raise SettlementIntegrityError("settlement ledger is not canonical")
            self._validate(document)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, SettlementIntegrityError):
                raise
            raise SettlementIntegrityError(
                "settlement ledger content is invalid"
            ) from exc
        return document

    @staticmethod
    def _validate(document: Mapping[str, object]) -> None:
        tickets = document["tickets"]
        assert isinstance(tickets, list)
        prior_ticket = ""
        for raw in tickets:
            if not isinstance(raw, dict) or set(raw) != {
                "fragments",
                "last_fragment_sha256",
                "requests",
                "seen_mask",
                "seen_values",
                "series_id",
                "target_dim",
                "ticket_id",
                "timestamps_ns",
            }:
                raise SettlementIntegrityError("ticket settlement entry is invalid")
            ticket_id = _ticket_id(raw["ticket_id"])
            if ticket_id <= prior_ticket:
                raise SettlementIntegrityError("ticket entries are not uniquely sorted")
            prior_ticket = ticket_id
            timestamps = raw["timestamps_ns"]
            target_dim = raw["target_dim"]
            if (
                not isinstance(raw["series_id"], str)
                or not raw["series_id"]
                or not isinstance(timestamps, list)
                or not timestamps
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in timestamps
                )
                or any(
                    right <= left
                    for left, right in zip(timestamps, timestamps[1:], strict=False)
                )
                or isinstance(target_dim, bool)
                or not isinstance(target_dim, int)
                or target_dim <= 0
            ):
                raise SettlementIntegrityError("ticket forecast contract is invalid")
            shape = (len(timestamps), target_dim)
            mask = np.asarray(raw["seen_mask"], dtype=np.bool_)
            values = np.asarray(raw["seen_values"], dtype=np.float32)
            if (
                mask.shape != shape
                or values.shape != shape
                or not np.isfinite(values).all()
            ):
                raise SettlementIntegrityError(
                    "ticket settled target matrix is invalid"
                )
            if bool((~mask & (values != 0.0)).any()):
                raise SettlementIntegrityError(
                    "unobserved ticket values are not canonical"
                )
            fragments = raw["fragments"]
            requests = raw["requests"]
            if (
                not isinstance(fragments, list)
                or not isinstance(requests, list)
                or len(fragments) > MAX_FRAGMENTS_PER_TICKET
                or len(requests) > MAX_FRAGMENTS_PER_TICKET
            ):
                raise SettlementIntegrityError("ticket settlement indexes are invalid")
            by_fragment: set[str] = set()
            prior_fragment = ""
            for fragment in fragments:
                if not isinstance(fragment, dict) or set(fragment) != {
                    "fragment_sha256",
                    "mask",
                    "values",
                }:
                    raise SettlementIntegrityError("fragment record is invalid")
                digest = require_sha256(
                    fragment["fragment_sha256"], field="fragment_sha256"
                )
                if digest <= prior_fragment:
                    raise SettlementIntegrityError(
                        "fragment records are not uniquely sorted"
                    )
                prior_fragment = digest
                fragment_mask = np.asarray(fragment["mask"], dtype=np.bool_)
                fragment_values = np.asarray(fragment["values"], dtype=np.float32)
                if (
                    fragment_mask.shape != shape
                    or fragment_values.shape != shape
                    or not fragment_mask.any()
                    or not np.isfinite(fragment_values).all()
                    or bool((~fragment_mask & (fragment_values != 0.0)).any())
                ):
                    raise SettlementIntegrityError("fragment target matrix is invalid")
                target = ForecastTarget(
                    series_id=raw["series_id"],
                    timestamps=np.asarray(timestamps, dtype="datetime64[ns]"),
                    values=fragment_values,
                    mask=fragment_mask,
                )
                if _fragment_sha256(ticket_id, target) != digest:
                    raise SettlementIntegrityError("fragment target hash is invalid")
                by_fragment.add(digest)
            last = raw["last_fragment_sha256"]
            if last is not None and (
                require_sha256(last, field="last_fragment_sha256") not in by_fragment
            ):
                raise SettlementIntegrityError("last fragment is absent")
            prior_request = ""
            for request in requests:
                if not isinstance(request, dict) or set(request) != {
                    "fragment_sha256",
                    "request_sha256",
                }:
                    raise SettlementIntegrityError("request record is invalid")
                request_sha = require_sha256(
                    request["request_sha256"], field="request_sha256"
                )
                fragment_sha = require_sha256(
                    request["fragment_sha256"], field="fragment_sha256"
                )
                if request_sha <= prior_request or fragment_sha not in by_fragment:
                    raise SettlementIntegrityError("request index is invalid")
                prior_request = request_sha

    @contextmanager
    def transaction(self) -> Iterator[SettlementLedger]:
        """Lock, reload, execute one settlement transition, then release."""

        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                self._document = self._load()
                yield self
            finally:
                self._document = None
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _tickets(self) -> list[dict[str, object]]:
        if self._document is None:
            raise RuntimeError("settlement ledger operation requires a transaction")
        tickets = self._document["tickets"]
        assert isinstance(tickets, list)
        return tickets

    def _entry(
        self,
        ticket_id: str,
        target: ForecastTarget,
        *,
        create: bool,
    ) -> dict[str, object] | None:
        timestamps = _timestamps_ns(target)
        for raw in self._tickets():
            if raw["ticket_id"] != ticket_id:
                continue
            if (
                raw["series_id"] != target.series_id
                or raw["timestamps_ns"] != timestamps
                or raw["target_dim"] != target.target_dim
            ):
                raise SettlementConflictError(
                    "ticket_id is bound to another forecast target contract"
                )
            return raw
        if not create:
            return None
        entry: dict[str, object] = {
            "fragments": [],
            "last_fragment_sha256": None,
            "requests": [],
            "seen_mask": np.zeros_like(target.mask, dtype=np.bool_).tolist(),
            "seen_values": np.zeros_like(target.values, dtype=np.float32).tolist(),
            "series_id": target.series_id,
            "target_dim": target.target_dim,
            "ticket_id": ticket_id,
            "timestamps_ns": timestamps,
        }
        self._tickets().append(entry)
        self._tickets().sort(key=lambda value: str(value["ticket_id"]))
        return entry

    @staticmethod
    def _fragment_target(
        entry: Mapping[str, object], fragment_sha256: str
    ) -> ForecastTarget:
        fragments = entry["fragments"]
        assert isinstance(fragments, list)
        for fragment in fragments:
            assert isinstance(fragment, dict)
            if fragment["fragment_sha256"] == fragment_sha256:
                return ForecastTarget(
                    series_id=entry["series_id"],
                    timestamps=np.asarray(
                        entry["timestamps_ns"], dtype="datetime64[ns]"
                    ),
                    values=fragment["values"],
                    mask=fragment["mask"],
                )
        raise SettlementIntegrityError("request refers to an absent fragment")

    def prepare(self, ticket_id: str, target: ForecastTarget) -> SettlementFragment:
        """Return only cells not previously committed for this ticket."""

        selected_id = _ticket_id(ticket_id)
        if not isinstance(target, ForecastTarget):
            raise TypeError("target must be ForecastTarget")
        if not bool(target.mask.any()):
            raise ValueError("settlement target observes no values")
        request_sha = _request_sha256(selected_id, target)
        entry = self._entry(selected_id, target, create=True)
        assert entry is not None
        requests = entry["requests"]
        assert isinstance(requests, list)
        for request in requests:
            assert isinstance(request, dict)
            if request["request_sha256"] == request_sha:
                fragment_sha = require_sha256(
                    request["fragment_sha256"], field="fragment_sha256"
                )
                return SettlementFragment(
                    ticket_id=selected_id,
                    request_sha256=request_sha,
                    fragment_sha256=fragment_sha,
                    target=self._fragment_target(entry, fragment_sha),
                    replayed=True,
                )

        seen_mask = np.asarray(entry["seen_mask"], dtype=np.bool_)
        seen_values = np.asarray(entry["seen_values"], dtype=np.float32)
        overlap = seen_mask & target.mask
        if bool((seen_values[overlap] != target.values[overlap]).any()):
            raise SettlementConflictError(
                "settlement changes a target value observed by an earlier fragment"
            )
        delta_mask = target.mask & ~seen_mask
        if not bool(delta_mask.any()):
            last = entry["last_fragment_sha256"]
            if last is None:
                raise SettlementIntegrityError("settled mask has no fragment history")
            fragment_sha = require_sha256(last, field="last_fragment_sha256")
            fragment_target = self._fragment_target(entry, fragment_sha)
            return SettlementFragment(
                ticket_id=selected_id,
                request_sha256=request_sha,
                fragment_sha256=fragment_sha,
                target=fragment_target,
                replayed=True,
            )
        fragment_target = ForecastTarget(
            series_id=target.series_id,
            timestamps=target.timestamps,
            values=target.values,
            mask=delta_mask,
        )
        fragment_sha = _fragment_sha256(selected_id, fragment_target)
        return SettlementFragment(
            ticket_id=selected_id,
            request_sha256=request_sha,
            fragment_sha256=fragment_sha,
            target=fragment_target,
            replayed=False,
        )

    def commit(self, fragment: SettlementFragment) -> None:
        """Atomically publish one prepared fragment after all consumers succeed."""

        if not isinstance(fragment, SettlementFragment):
            raise TypeError("fragment must be SettlementFragment")
        entry = self._entry(fragment.ticket_id, fragment.target, create=True)
        assert entry is not None
        requests = entry["requests"]
        fragments = entry["fragments"]
        assert isinstance(requests, list) and isinstance(fragments, list)
        for request in requests:
            assert isinstance(request, dict)
            if request["request_sha256"] == fragment.request_sha256:
                if request["fragment_sha256"] != fragment.fragment_sha256:
                    raise SettlementConflictError(
                        "settlement request is bound to another fragment"
                    )
                return

        existing = next(
            (
                value
                for value in fragments
                if isinstance(value, dict)
                and value["fragment_sha256"] == fragment.fragment_sha256
            ),
            None,
        )
        record = {
            "fragment_sha256": fragment.fragment_sha256,
            "mask": fragment.target.mask.tolist(),
            "values": fragment.target.values.tolist(),
        }
        if existing is not None and existing != record:
            raise SettlementConflictError("fragment identity has different evidence")
        if existing is None:
            if len(fragments) >= MAX_FRAGMENTS_PER_TICKET:
                raise SettlementLedgerError("ticket fragment capacity is exhausted")
            fragments.append(record)
            fragments.sort(key=lambda value: str(value["fragment_sha256"]))

        seen_mask = np.asarray(entry["seen_mask"], dtype=np.bool_)
        seen_values = np.asarray(entry["seen_values"], dtype=np.float32)
        overlap = seen_mask & fragment.target.mask
        if bool((seen_values[overlap] != fragment.target.values[overlap]).any()):
            raise SettlementConflictError("fragment conflicts with committed evidence")
        new_mask = fragment.target.mask & ~seen_mask
        seen_values[new_mask] = fragment.target.values[new_mask]
        seen_mask |= fragment.target.mask
        entry["seen_mask"] = seen_mask.tolist()
        entry["seen_values"] = seen_values.tolist()
        entry["last_fragment_sha256"] = fragment.fragment_sha256
        requests.append(
            {
                "fragment_sha256": fragment.fragment_sha256,
                "request_sha256": fragment.request_sha256,
            }
        )
        requests.sort(key=lambda value: str(value["request_sha256"]))
        assert self._document is not None
        payload = canonical_json_bytes(self._document)
        if len(payload) > MAX_STATE_BYTES:
            raise SettlementLedgerError("settlement ledger exceeds its size bound")
        _atomic_write(self.path, payload)


__all__ = [
    "SETTLEMENT_FRAGMENT_SCHEMA",
    "SETTLEMENT_LEDGER_SCHEMA",
    "SettlementConflictError",
    "SettlementFragment",
    "SettlementIntegrityError",
    "SettlementLedger",
    "SettlementLedgerError",
]
