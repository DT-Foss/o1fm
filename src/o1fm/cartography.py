"""Autonomous forecast cartography with authenticated online evidence.

The cartographer decides *where to measure next*.  It never executes a model,
reads a future target, or invents a score on behalf of an executor.  Callers
register immutable probe targets and settle them with realized evidence after
an external executor has completed the work.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from .identity import canonical_json_bytes, require_sha256, sha256_json

TARGET_SCHEMA = "o1fm.forecast-probe-target/v1"
EVIDENCE_SCHEMA = "o1fm.forecast-probe-evidence/v1"
RECEIPT_SCHEMA = "o1fm.forecast-probe-receipt/v1"
EVENT_SCHEMA = "o1fm.forecast-cartography-event/v1"
STATE_SCHEMA = "o1fm.forecast-cartographer/v1"
CONFIG_SCHEMA = "o1fm.forecast-cartographer-config/v1"
BUDGET_SCHEMA = "o1fm.forecast-cartographer-budgets/v1"
ZERO_SHA256 = "0" * 64

MAX_COUNTER = (1 << 63) - 1
MAX_TEXT_BYTES = 4096
MAX_TARGETS = 1_000_000
MAX_ACTIONS = 1024
MAX_TARGET_INDICES = 1_000_000
MAX_FEATURE_DIMENSION = 16_384
MAX_EVENT_BYTES = 2 * 1024 * 1024
MAX_STATE_BYTES = 128 * 1024 * 1024

_EVENT_NAME = re.compile(r"([0-9]{20})-([0-9a-f]{64})\.json\Z")
_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}


class CartographyError(RuntimeError):
    """The requested cartography operation cannot be completed."""


class CartographyIntegrityError(CartographyError):
    """Persistent cartography state is malformed or unauthenticated."""


class ProbeConflictError(CartographyError):
    """A settled probe was replayed with different evidence."""


class CartographyBudgetError(CartographyError):
    """A probe would exceed its declared or global work budget."""


def _root_lock(root: Path) -> threading.RLock:
    key = str(root.resolve())
    with _LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.RLock())


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > MAX_TEXT_BYTES
    ):
        raise ValueError(f"{field} must be canonical bounded NFC text")
    return value


def _finite(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return 0.0 if result == 0.0 else result


def _positive_float(value: object, *, field: str) -> float:
    result = _finite(value, field=field, minimum=0.0)
    if result == 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _uint(
    value: object,
    *,
    field: str,
    positive: bool = False,
    maximum: int = MAX_COUNTER,
) -> int:
    lower = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= maximum
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return value


def _float_hex(value: float) -> str:
    return _finite(value, field="float").hex()


def _parse_hex(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, str) or len(value) > 32:
        raise CartographyIntegrityError(f"{field} is not a canonical float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise CartographyIntegrityError(f"{field} is not a canonical float") from exc
    result = _finite(result, field=field, minimum=minimum, maximum=maximum)
    if result.hex() != value:
        raise CartographyIntegrityError(f"{field} is not a canonical float")
    return result


def _record_fields(value: object, fields: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CartographyIntegrityError(f"{name} fields are invalid")
    return value


def _read_canonical(path: Path, *, maximum: int) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CartographyIntegrityError(f"cannot read {path.name}") from exc
    if len(raw) > maximum:
        raise CartographyIntegrityError(f"{path.name} exceeds its byte bound")
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CartographyIntegrityError(f"{path.name} is not valid JSON") from exc
    if not isinstance(document, Mapping) or canonical_json_bytes(document) != raw:
        raise CartographyIntegrityError(f"{path.name} is not canonical JSON")
    return document


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
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


def _append_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CartographyIntegrityError("event sequence already exists") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ForecastProbeTarget:
    """One exact, immutable forecast measurement contract."""

    series_id: str
    dataset_identity_sha256: str
    model_identity_sha256: str
    graph_topology_sha256: str
    regime: str
    horizon: int
    target_indices: tuple[int, ...]
    action_inventory: tuple[str, ...]
    intervention_identity_sha256: str = ZERO_SHA256
    seed: int = 0
    expected_work: float = 1.0
    expected_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_id", _text(self.series_id, field="series_id"))
        object.__setattr__(self, "regime", _text(self.regime, field="regime"))
        for field in (
            "dataset_identity_sha256",
            "model_identity_sha256",
            "graph_topology_sha256",
            "intervention_identity_sha256",
        ):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        object.__setattr__(
            self,
            "horizon",
            _uint(self.horizon, field="horizon", positive=True),
        )
        if isinstance(self.target_indices, (str, bytes)):
            raise ValueError("target_indices must be an integer sequence")
        targets = tuple(
            sorted(
                {
                    _uint(value, field="target index")
                    for value in self.target_indices
                }
            )
        )
        if not targets or len(targets) > MAX_TARGET_INDICES:
            raise ValueError("target_indices must be non-empty and bounded")
        object.__setattr__(self, "target_indices", targets)
        if isinstance(self.action_inventory, (str, bytes)):
            raise ValueError("action_inventory must be a text sequence")
        actions = tuple(
            sorted({_text(value, field="action") for value in self.action_inventory})
        )
        if not actions or len(actions) > MAX_ACTIONS:
            raise ValueError("action_inventory must be non-empty and bounded")
        object.__setattr__(self, "action_inventory", actions)
        object.__setattr__(self, "seed", _uint(self.seed, field="seed"))
        object.__setattr__(
            self,
            "expected_work",
            _positive_float(self.expected_work, field="expected_work"),
        )
        object.__setattr__(
            self,
            "expected_bytes",
            _uint(self.expected_bytes, field="expected_bytes"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "action_inventory": list(self.action_inventory),
            "dataset_identity_sha256": self.dataset_identity_sha256,
            "expected_bytes": self.expected_bytes,
            "expected_work": self.expected_work.hex(),
            "graph_topology_sha256": self.graph_topology_sha256,
            "horizon": self.horizon,
            "intervention_identity_sha256": self.intervention_identity_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "regime": self.regime,
            "schema": TARGET_SCHEMA,
            "seed": self.seed,
            "series_id": self.series_id,
            "target_indices": list(self.target_indices),
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.to_record())

    @property
    def sha256(self) -> str:
        return self.identity_sha256

    @classmethod
    def from_record(cls, value: object) -> ForecastProbeTarget:
        fields = {
            "action_inventory",
            "dataset_identity_sha256",
            "expected_bytes",
            "expected_work",
            "graph_topology_sha256",
            "horizon",
            "intervention_identity_sha256",
            "model_identity_sha256",
            "regime",
            "schema",
            "seed",
            "series_id",
            "target_indices",
        }
        record = _record_fields(value, fields, name="probe target")
        if record["schema"] != TARGET_SCHEMA:
            raise CartographyIntegrityError("probe target schema is unsupported")
        try:
            result = cls(
                series_id=record["series_id"],
                dataset_identity_sha256=record["dataset_identity_sha256"],
                model_identity_sha256=record["model_identity_sha256"],
                graph_topology_sha256=record["graph_topology_sha256"],
                regime=record["regime"],
                horizon=record["horizon"],
                target_indices=tuple(record["target_indices"]),
                action_inventory=tuple(record["action_inventory"]),
                intervention_identity_sha256=record[
                    "intervention_identity_sha256"
                ],
                seed=record["seed"],
                expected_work=_parse_hex(
                    record["expected_work"],
                    field="expected_work",
                    minimum=0.0,
                ),
                expected_bytes=record["expected_bytes"],
            )
        except (TypeError, ValueError) as exc:
            raise CartographyIntegrityError("probe target is invalid") from exc
        if result.to_record() != dict(record):
            raise CartographyIntegrityError("probe target is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    """Realized forecast evidence returned by an external probe executor."""

    target_sha256: str
    coverage_gap: float
    pinball: float
    mae: float
    joint_miss: float
    o1_surprise: float
    learning_progress: float
    route_w1: float
    observed_values: int
    work_cost: float
    result_sha256: str
    atlas_sha256: str
    started_at: float
    completed_at: float

    def __post_init__(self) -> None:
        for field in ("target_sha256", "result_sha256", "atlas_sha256"):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        object.__setattr__(
            self,
            "coverage_gap",
            _finite(self.coverage_gap, field="coverage_gap", minimum=-1.0, maximum=1.0),
        )
        for field in ("pinball", "mae", "o1_surprise", "route_w1", "work_cost"):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field=field, minimum=0.0),
            )
        object.__setattr__(
            self,
            "joint_miss",
            _finite(self.joint_miss, field="joint_miss", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "learning_progress",
            _finite(self.learning_progress, field="learning_progress"),
        )
        object.__setattr__(
            self,
            "observed_values",
            _uint(self.observed_values, field="observed_values", positive=True),
        )
        started = _finite(self.started_at, field="started_at", minimum=0.0)
        completed = _finite(self.completed_at, field="completed_at", minimum=0.0)
        if completed < started:
            raise ValueError("completed_at must not precede started_at")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "completed_at", completed)

    def to_record(self) -> dict[str, object]:
        return {
            "atlas_sha256": self.atlas_sha256,
            "completed_at": self.completed_at.hex(),
            "coverage_gap": self.coverage_gap.hex(),
            "joint_miss": self.joint_miss.hex(),
            "learning_progress": self.learning_progress.hex(),
            "mae": self.mae.hex(),
            "o1_surprise": self.o1_surprise.hex(),
            "observed_values": self.observed_values,
            "pinball": self.pinball.hex(),
            "result_sha256": self.result_sha256,
            "route_w1": self.route_w1.hex(),
            "schema": EVIDENCE_SCHEMA,
            "started_at": self.started_at.hex(),
            "target_sha256": self.target_sha256,
            "work_cost": self.work_cost.hex(),
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.to_record())

    @property
    def sha256(self) -> str:
        return self.identity_sha256

    @classmethod
    def from_record(cls, value: object) -> ProbeEvidence:
        fields = {
            "atlas_sha256",
            "completed_at",
            "coverage_gap",
            "joint_miss",
            "learning_progress",
            "mae",
            "o1_surprise",
            "observed_values",
            "pinball",
            "result_sha256",
            "route_w1",
            "schema",
            "started_at",
            "target_sha256",
            "work_cost",
        }
        record = _record_fields(value, fields, name="probe evidence")
        if record["schema"] != EVIDENCE_SCHEMA:
            raise CartographyIntegrityError("probe evidence schema is unsupported")
        try:
            result = cls(
                target_sha256=record["target_sha256"],
                coverage_gap=_parse_hex(
                    record["coverage_gap"],
                    field="coverage_gap",
                    minimum=-1.0,
                    maximum=1.0,
                ),
                pinball=_parse_hex(record["pinball"], field="pinball", minimum=0.0),
                mae=_parse_hex(record["mae"], field="mae", minimum=0.0),
                joint_miss=_parse_hex(
                    record["joint_miss"],
                    field="joint_miss",
                    minimum=0.0,
                    maximum=1.0,
                ),
                o1_surprise=_parse_hex(
                    record["o1_surprise"],
                    field="o1_surprise",
                    minimum=0.0,
                ),
                learning_progress=_parse_hex(
                    record["learning_progress"],
                    field="learning_progress",
                ),
                route_w1=_parse_hex(
                    record["route_w1"], field="route_w1", minimum=0.0
                ),
                observed_values=record["observed_values"],
                work_cost=_parse_hex(
                    record["work_cost"], field="work_cost", minimum=0.0
                ),
                result_sha256=record["result_sha256"],
                atlas_sha256=record["atlas_sha256"],
                started_at=_parse_hex(
                    record["started_at"], field="started_at", minimum=0.0
                ),
                completed_at=_parse_hex(
                    record["completed_at"], field="completed_at", minimum=0.0
                ),
            )
        except (TypeError, ValueError) as exc:
            raise CartographyIntegrityError("probe evidence is invalid") from exc
        if result.to_record() != dict(record):
            raise CartographyIntegrityError("probe evidence is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ProbeReceipt:
    """One authenticated link in the append-only settlement chain."""

    sequence: int
    target_sha256: str
    evidence: ProbeEvidence
    previous_receipt_sha256: str
    priority_at_settlement: float
    information_gain: float
    committed_at: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _uint(self.sequence, field="sequence", positive=True),
        )
        object.__setattr__(
            self,
            "target_sha256",
            require_sha256(self.target_sha256, field="target_sha256"),
        )
        if not isinstance(self.evidence, ProbeEvidence):
            raise TypeError("evidence must be ProbeEvidence")
        if self.evidence.target_sha256 != self.target_sha256:
            raise ValueError("receipt and evidence target identities differ")
        object.__setattr__(
            self,
            "previous_receipt_sha256",
            require_sha256(
                self.previous_receipt_sha256,
                field="previous_receipt_sha256",
            ),
        )
        object.__setattr__(
            self,
            "priority_at_settlement",
            _finite(
                self.priority_at_settlement,
                field="priority_at_settlement",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "information_gain",
            _finite(self.information_gain, field="information_gain", minimum=0.0),
        )
        object.__setattr__(
            self,
            "committed_at",
            _finite(self.committed_at, field="committed_at", minimum=0.0),
        )
        if self.committed_at < self.evidence.completed_at:
            raise ValueError("committed_at must not precede evidence completion")

    def to_record(self) -> dict[str, object]:
        return {
            "committed_at": self.committed_at.hex(),
            "evidence": self.evidence.to_record(),
            "information_gain": self.information_gain.hex(),
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "priority_at_settlement": self.priority_at_settlement.hex(),
            "schema": RECEIPT_SCHEMA,
            "sequence": self.sequence,
            "target_sha256": self.target_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return sha256_json(self.to_record())

    @property
    def sha256(self) -> str:
        return self.identity_sha256

    @property
    def coverage_gap(self) -> float:
        return self.evidence.coverage_gap

    @property
    def pinball(self) -> float:
        return self.evidence.pinball

    @property
    def mae(self) -> float:
        return self.evidence.mae

    @property
    def joint_miss(self) -> float:
        return self.evidence.joint_miss

    @property
    def o1_surprise(self) -> float:
        return self.evidence.o1_surprise

    @property
    def learning_progress(self) -> float:
        return self.evidence.learning_progress

    @property
    def route_w1(self) -> float:
        return self.evidence.route_w1

    @property
    def observed_values(self) -> int:
        return self.evidence.observed_values

    @property
    def work_cost(self) -> float:
        return self.evidence.work_cost

    @property
    def result_sha256(self) -> str:
        return self.evidence.result_sha256

    @property
    def atlas_sha256(self) -> str:
        return self.evidence.atlas_sha256

    @property
    def started_at(self) -> float:
        return self.evidence.started_at

    @property
    def completed_at(self) -> float:
        return self.evidence.completed_at

    @classmethod
    def from_record(cls, value: object) -> ProbeReceipt:
        fields = {
            "committed_at",
            "evidence",
            "information_gain",
            "previous_receipt_sha256",
            "priority_at_settlement",
            "schema",
            "sequence",
            "target_sha256",
        }
        record = _record_fields(value, fields, name="probe receipt")
        if record["schema"] != RECEIPT_SCHEMA:
            raise CartographyIntegrityError("probe receipt schema is unsupported")
        try:
            result = cls(
                sequence=record["sequence"],
                target_sha256=record["target_sha256"],
                evidence=ProbeEvidence.from_record(record["evidence"]),
                previous_receipt_sha256=record["previous_receipt_sha256"],
                priority_at_settlement=_parse_hex(
                    record["priority_at_settlement"],
                    field="priority_at_settlement",
                    minimum=0.0,
                ),
                information_gain=_parse_hex(
                    record["information_gain"],
                    field="information_gain",
                    minimum=0.0,
                ),
                committed_at=_parse_hex(
                    record["committed_at"],
                    field="committed_at",
                    minimum=0.0,
                ),
            )
        except (TypeError, ValueError) as exc:
            raise CartographyIntegrityError("probe receipt is invalid") from exc
        if result.to_record() != dict(record):
            raise CartographyIntegrityError("probe receipt is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CartographerBudgets:
    """Lifetime work and stored-byte limits for one cartographer ledger."""

    work: float = 1e18
    bytes: int = MAX_COUNTER

    def __post_init__(self) -> None:
        object.__setattr__(self, "work", _finite(self.work, field="work", minimum=0.0))
        object.__setattr__(self, "bytes", _uint(self.bytes, field="bytes"))

    def to_record(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "schema": BUDGET_SCHEMA,
            "work": self.work.hex(),
        }

    @classmethod
    def from_record(cls, value: object) -> CartographerBudgets:
        record = _record_fields(value, {"bytes", "schema", "work"}, name="budgets")
        if record["schema"] != BUDGET_SCHEMA:
            raise CartographyIntegrityError("budget schema is unsupported")
        try:
            result = cls(
                work=_parse_hex(record["work"], field="work", minimum=0.0),
                bytes=record["bytes"],
            )
        except (TypeError, ValueError) as exc:
            raise CartographyIntegrityError("budgets are invalid") from exc
        if result.to_record() != dict(record):
            raise CartographyIntegrityError("budgets are not canonical")
        return result


@dataclass(frozen=True, slots=True)
class CartographerConfig:
    """Pinned value-of-information and acquisition policy."""

    base_value: float = 0.05
    coverage_weight: float = 4.0
    joint_miss_weight: float = 3.0
    surprise_weight: float = 2.0
    unexplored_weight: float = 5.0
    route_w1_weight: float = 1.0
    learning_progress_weight: float = 1.5
    work_cost_scale: float = 1.0
    byte_cost_scale: float = 1e-9
    low_learning_threshold: float = 0.02
    low_learning_decay: float = 1.0
    d_optimal_ridge: float = 1e-6
    max_feature_dimension: int = 1024
    max_feature_absolute: float = 1e6

    def __post_init__(self) -> None:
        for field in (
            "base_value",
            "coverage_weight",
            "joint_miss_weight",
            "surprise_weight",
            "unexplored_weight",
            "route_w1_weight",
            "learning_progress_weight",
            "work_cost_scale",
            "byte_cost_scale",
            "low_learning_threshold",
            "low_learning_decay",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field=field, minimum=0.0),
            )
        object.__setattr__(
            self,
            "d_optimal_ridge",
            _positive_float(self.d_optimal_ridge, field="d_optimal_ridge"),
        )
        object.__setattr__(
            self,
            "max_feature_dimension",
            _uint(
                self.max_feature_dimension,
                field="max_feature_dimension",
                positive=True,
                maximum=MAX_FEATURE_DIMENSION,
            ),
        )
        object.__setattr__(
            self,
            "max_feature_absolute",
            _positive_float(self.max_feature_absolute, field="max_feature_absolute"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "base_value": self.base_value.hex(),
            "byte_cost_scale": self.byte_cost_scale.hex(),
            "coverage_weight": self.coverage_weight.hex(),
            "d_optimal_ridge": self.d_optimal_ridge.hex(),
            "joint_miss_weight": self.joint_miss_weight.hex(),
            "learning_progress_weight": self.learning_progress_weight.hex(),
            "low_learning_decay": self.low_learning_decay.hex(),
            "low_learning_threshold": self.low_learning_threshold.hex(),
            "max_feature_absolute": self.max_feature_absolute.hex(),
            "max_feature_dimension": self.max_feature_dimension,
            "route_w1_weight": self.route_w1_weight.hex(),
            "schema": CONFIG_SCHEMA,
            "surprise_weight": self.surprise_weight.hex(),
            "unexplored_weight": self.unexplored_weight.hex(),
            "work_cost_scale": self.work_cost_scale.hex(),
        }

    @classmethod
    def from_record(cls, value: object) -> CartographerConfig:
        float_fields = {
            "base_value",
            "byte_cost_scale",
            "coverage_weight",
            "d_optimal_ridge",
            "joint_miss_weight",
            "learning_progress_weight",
            "low_learning_decay",
            "low_learning_threshold",
            "max_feature_absolute",
            "route_w1_weight",
            "surprise_weight",
            "unexplored_weight",
            "work_cost_scale",
        }
        fields = float_fields | {"max_feature_dimension", "schema"}
        record = _record_fields(value, fields, name="cartographer config")
        if record["schema"] != CONFIG_SCHEMA:
            raise CartographyIntegrityError("cartographer config schema is unsupported")
        kwargs: dict[str, object] = {
            field: _parse_hex(record[field], field=field, minimum=0.0)
            for field in float_fields
        }
        kwargs["max_feature_dimension"] = record["max_feature_dimension"]
        try:
            result = cls(**kwargs)
        except (TypeError, ValueError) as exc:
            raise CartographyIntegrityError("cartographer config is invalid") from exc
        if result.to_record() != dict(record):
            raise CartographyIntegrityError("cartographer config is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ProbePriority:
    """Inspectable productive priority for one pending probe."""

    target_sha256: str
    value: float
    value_density: float
    priority: float
    unexplored_fraction: float
    related_receipts: int
    low_learning_decay: float

    def __post_init__(self) -> None:
        require_sha256(self.target_sha256, field="target_sha256")
        for field in (
            "value",
            "value_density",
            "priority",
            "unexplored_fraction",
            "low_learning_decay",
        ):
            _finite(getattr(self, field), field=field, minimum=0.0)
        _uint(self.related_receipts, field="related_receipts")


class BackgroundRun:
    """Minimal future for one serial background cartography run."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._receipts: tuple[ProbeReceipt, ...] | None = None
        self._error: BaseException | None = None

    def _finish(
        self,
        receipts: tuple[ProbeReceipt, ...] | None,
        error: BaseException | None,
    ) -> None:
        self._receipts = receipts
        self._error = error
        self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def result(self, timeout: float | None = None) -> tuple[ProbeReceipt, ...]:
        if not self._done.wait(timeout):
            raise TimeoutError("background cartography run is still active")
        if self._error is not None:
            raise self._error
        assert self._receipts is not None
        return self._receipts

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if not self._done.wait(timeout):
            raise TimeoutError("background cartography run is still active")
        return self._error


ProbeExecutor = Callable[[ForecastProbeTarget], ProbeEvidence]


class ForecastCartographer:
    """Persistent autonomous frontier over forecast-native probe targets."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        capacity: int | None = None,
        budgets: CartographerBudgets | None = None,
        work_budget: float | None = None,
        byte_budget: int | None = None,
        config: CartographerConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.state_path = self.root / "cartographer.json"
        self.event_root = self.root / "evidence"
        self._lock = _root_lock(self.root)
        self._targets: dict[str, ForecastProbeTarget] = {}
        self._receipts: list[ProbeReceipt] = []
        self._settled: dict[str, ProbeReceipt] = {}
        self._inflight: set[str] = set()
        self._background_queue: queue.Queue[
            tuple[ProbeExecutor, int, float | None, int | None, BackgroundRun] | None
        ] = queue.Queue()
        self._background_thread: threading.Thread | None = None
        self._background_closed = False

        if budgets is not None and (work_budget is not None or byte_budget is not None):
            raise ValueError("pass budgets or individual budget overrides, not both")
        if budgets is not None and not isinstance(budgets, CartographerBudgets):
            raise TypeError("budgets must be CartographerBudgets")
        if config is not None and not isinstance(config, CartographerConfig):
            raise TypeError("config must be CartographerConfig")

        with self._lock:
            if self.state_path.exists():
                self._load()
                if capacity is not None and self.capacity != _uint(
                    capacity,
                    field="capacity",
                    positive=True,
                    maximum=MAX_TARGETS,
                ):
                    raise CartographyIntegrityError("persisted capacity differs")
                requested_budgets = budgets
                if work_budget is not None or byte_budget is not None:
                    requested_budgets = CartographerBudgets(
                        work=self.budgets.work
                        if work_budget is None
                        else work_budget,
                        bytes=self.budgets.bytes
                        if byte_budget is None
                        else byte_budget,
                    )
                if requested_budgets is not None and requested_budgets != self.budgets:
                    raise CartographyIntegrityError("persisted budgets differ")
                if config is not None and config != self.config:
                    raise CartographyIntegrityError("persisted config differs")
            else:
                self.capacity = _uint(
                    100_000 if capacity is None else capacity,
                    field="capacity",
                    positive=True,
                    maximum=MAX_TARGETS,
                )
                if budgets is None:
                    budgets = CartographerBudgets(
                        work=1e18 if work_budget is None else work_budget,
                        bytes=MAX_COUNTER if byte_budget is None else byte_budget,
                    )
                self.budgets = budgets
                self.config = CartographerConfig() if config is None else config
                if self.event_root.exists() and any(self.event_root.iterdir()):
                    raise CartographyIntegrityError(
                        "evidence exists without its target checkpoint"
                    )
                self.root.mkdir(parents=True, exist_ok=True)
                self.event_root.mkdir(parents=True, exist_ok=True)
                self._write_checkpoint()

    @property
    def targets(self) -> tuple[ForecastProbeTarget, ...]:
        with self._lock:
            return tuple(self._targets[key] for key in sorted(self._targets))

    @property
    def receipts(self) -> tuple[ProbeReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    @property
    def head_receipt_sha256(self) -> str:
        with self._lock:
            return self._receipts[-1].sha256 if self._receipts else ZERO_SHA256

    def _state_body(self) -> dict[str, object]:
        return {
            "budgets": self.budgets.to_record(),
            "capacity": self.capacity,
            "completed_target_sha256s": sorted(self._settled),
            "config": self.config.to_record(),
            "head_receipt_sha256": (
                self._receipts[-1].sha256 if self._receipts else ZERO_SHA256
            ),
            "sequence": len(self._receipts),
            "targets": [
                self._targets[key].to_record() for key in sorted(self._targets)
            ],
        }

    def _state_document(self) -> dict[str, object]:
        body = self._state_body()
        return {
            "body": body,
            "body_sha256": sha256_json(body),
            "schema": STATE_SCHEMA,
        }

    def _write_checkpoint(self) -> None:
        encoded = canonical_json_bytes(self._state_document())
        if len(encoded) > MAX_STATE_BYTES:
            raise CartographyIntegrityError("cartographer checkpoint exceeds its bound")
        _atomic_write(self.state_path, encoded)

    def checkpoint(self) -> str:
        """Atomically persist the current index and return its content identity."""

        with self._lock:
            document = self._state_document()
            self._write_checkpoint()
            return sha256_json(document)

    def _load(self) -> None:
        document = _read_canonical(self.state_path, maximum=MAX_STATE_BYTES)
        document = _record_fields(
            document,
            {"body", "body_sha256", "schema"},
            name="cartographer checkpoint",
        )
        if document["schema"] != STATE_SCHEMA or not isinstance(
            document["body"], Mapping
        ):
            raise CartographyIntegrityError("cartographer checkpoint schema is invalid")
        body = document["body"]
        if sha256_json(body) != document["body_sha256"]:
            raise CartographyIntegrityError("cartographer checkpoint hash differs")
        fields = {
            "budgets",
            "capacity",
            "completed_target_sha256s",
            "config",
            "head_receipt_sha256",
            "sequence",
            "targets",
        }
        body = _record_fields(body, fields, name="cartographer checkpoint body")
        try:
            self.capacity = _uint(
                body["capacity"],
                field="capacity",
                positive=True,
                maximum=MAX_TARGETS,
            )
            self.budgets = CartographerBudgets.from_record(body["budgets"])
            self.config = CartographerConfig.from_record(body["config"])
            raw_targets = body["targets"]
            if not isinstance(raw_targets, list) or len(raw_targets) > self.capacity:
                raise ValueError("target inventory is invalid")
            targets = [ForecastProbeTarget.from_record(item) for item in raw_targets]
        except (TypeError, ValueError) as exc:
            if isinstance(exc, CartographyIntegrityError):
                raise
            raise CartographyIntegrityError("cartographer checkpoint is invalid") from exc
        self._targets = {target.sha256: target for target in targets}
        if len(self._targets) != len(targets) or [
            target.sha256 for target in targets
        ] != sorted(self._targets):
            raise CartographyIntegrityError("target inventory is duplicated or unordered")

        self._receipts = []
        self._settled = {}
        self._replay_events()
        expected_completed = sorted(self._settled)
        if not isinstance(body["completed_target_sha256s"], list):
            raise CartographyIntegrityError("completed target inventory is invalid")
        checkpoint_sequence = _uint(body["sequence"], field="sequence")
        checkpoint_head = require_sha256(
            body["head_receipt_sha256"], field="head_receipt_sha256"
        )
        if checkpoint_sequence > len(self._receipts):
            raise CartographyIntegrityError("checkpoint is ahead of evidence history")
        if checkpoint_sequence == len(self._receipts):
            if (
                body["completed_target_sha256s"] != expected_completed
                or checkpoint_head
                != (self._receipts[-1].sha256 if self._receipts else ZERO_SHA256)
            ):
                raise CartographyIntegrityError("checkpoint and evidence history differ")
        else:
            # An immutable event won the crash race before the replace of state.
            self._write_checkpoint()

    def reload(self) -> None:
        """Verify and reload the checkpoint plus complete evidence history."""

        with self._lock:
            if self._inflight:
                raise CartographyError("cannot reload while probes are in flight")
            self._load()

    def _replay_events(self) -> None:
        if not self.event_root.exists():
            self.event_root.mkdir(parents=True, exist_ok=True)
            return
        paths = sorted(path for path in self.event_root.iterdir() if path.is_file())
        expected_sequence = 1
        previous = ZERO_SHA256
        for path in paths:
            match = _EVENT_NAME.fullmatch(path.name)
            if match is None or int(match.group(1)) != expected_sequence:
                raise CartographyIntegrityError("evidence history is not contiguous")
            document = _read_canonical(path, maximum=MAX_EVENT_BYTES)
            document = _record_fields(
                document,
                {"receipt", "receipt_sha256", "schema"},
                name="cartography event",
            )
            if document["schema"] != EVENT_SCHEMA:
                raise CartographyIntegrityError("cartography event schema is unsupported")
            receipt = ProbeReceipt.from_record(document["receipt"])
            if (
                receipt.sequence != expected_sequence
                or receipt.previous_receipt_sha256 != previous
                or receipt.sha256 != document["receipt_sha256"]
                or receipt.sha256 != match.group(2)
            ):
                raise CartographyIntegrityError("evidence hash chain is invalid")
            if receipt.target_sha256 not in self._targets:
                raise CartographyIntegrityError("evidence references an unknown target")
            if receipt.target_sha256 in self._settled:
                raise CartographyIntegrityError("target has multiple evidence events")
            target = self._targets[receipt.target_sha256]
            if receipt.work_cost > target.expected_work:
                raise CartographyIntegrityError("evidence exceeds declared target work")
            self._receipts.append(receipt)
            self._settled[receipt.target_sha256] = receipt
            expected_sequence += 1
            previous = receipt.sha256
        if self._total_work() > self.budgets.work or self._total_bytes() > self.budgets.bytes:
            raise CartographyIntegrityError("evidence history exceeds persisted budgets")

    def extend_targets(self, targets: Iterable[ForecastProbeTarget]) -> int:
        """Idempotently add immutable targets and return the number added."""

        materialized = tuple(targets)
        if any(not isinstance(target, ForecastProbeTarget) for target in materialized):
            raise TypeError("targets must contain ForecastProbeTarget values")
        incoming = {target.sha256: target for target in materialized}
        with self._lock:
            additions = {
                identity: target
                for identity, target in incoming.items()
                if identity not in self._targets
            }
            if len(self._targets) + len(additions) > self.capacity:
                raise CartographyError("target capacity would be exceeded")
            self._targets.update(additions)
            if additions:
                self._write_checkpoint()
            return len(additions)

    def extend(self, targets: Iterable[ForecastProbeTarget]) -> int:
        return self.extend_targets(targets)

    def add_target(self, target: ForecastProbeTarget) -> bool:
        return self.extend_targets((target,)) == 1

    @staticmethod
    def _context(target: ForecastProbeTarget) -> tuple[object, ...]:
        return (
            target.series_id,
            target.dataset_identity_sha256,
            target.model_identity_sha256,
            target.regime,
        )

    @classmethod
    def _cells(cls, target: ForecastProbeTarget) -> frozenset[tuple[object, ...]]:
        context = cls._context(target)
        return frozenset(
            [*(context + ("target", value) for value in target.target_indices)]
            + [context + ("horizon", target.horizon)]
            + [*(context + ("action", value) for value in target.action_inventory)]
            + [context + ("intervention", target.intervention_identity_sha256)]
        )

    def priority_breakdown(self, target: ForecastProbeTarget | str) -> ProbePriority:
        """Return the productive value estimate without consuming a job."""

        with self._lock:
            resolved = self._resolve_target(target)
            cells = self._cells(resolved)
            settled_cells: set[tuple[object, ...]] = set()
            related: list[ProbeReceipt] = []
            exact_support: list[ProbeReceipt] = []
            for receipt in self._receipts:
                historical = self._targets[receipt.target_sha256]
                historical_cells = self._cells(historical)
                settled_cells.update(historical_cells)
                if cells & historical_cells:
                    related.append(receipt)
                if cells == historical_cells:
                    exact_support.append(receipt)
            unexplored = sum(cell not in settled_cells for cell in cells) / len(cells)
            if related:
                inverse = 1.0 / len(related)
                coverage_gap = sum(abs(item.coverage_gap) for item in related) * inverse
                joint_miss = sum(item.joint_miss for item in related) * inverse
                surprise = sum(item.o1_surprise for item in related) * inverse
                route_w1 = sum(item.route_w1 for item in related) * inverse
                learning = (
                    sum(abs(item.learning_progress) for item in related) * inverse
                )
            else:
                coverage_gap = joint_miss = surprise = route_w1 = learning = 0.0
            decay = 1.0
            if exact_support:
                exact_learning = sum(
                    abs(item.learning_progress) for item in exact_support
                ) / len(exact_support)
                if exact_learning <= self.config.low_learning_threshold:
                    decay = 1.0 / (
                        1.0 + self.config.low_learning_decay * len(exact_support)
                    )
            value = decay * (
                self.config.base_value
                + self.config.coverage_weight * coverage_gap
                + self.config.joint_miss_weight * joint_miss
                + self.config.surprise_weight * surprise
                + self.config.unexplored_weight * unexplored
                + self.config.route_w1_weight * route_w1
                + self.config.learning_progress_weight * learning
            )
            cost = (
                1.0
                + self.config.work_cost_scale * resolved.expected_work
                + self.config.byte_cost_scale * resolved.expected_bytes
            )
            density = value / cost
            priority = value / math.sqrt(cost)
            return ProbePriority(
                target_sha256=resolved.sha256,
                value=value,
                value_density=density,
                priority=priority,
                unexplored_fraction=unexplored,
                related_receipts=len(related),
                low_learning_decay=decay,
            )

    def priority(self, target: ForecastProbeTarget | str) -> float:
        return self.priority_breakdown(target).priority

    def value_density(self, target: ForecastProbeTarget | str) -> float:
        return self.priority_breakdown(target).value_density

    def _resolve_target(self, target: ForecastProbeTarget | str) -> ForecastProbeTarget:
        identity = target.sha256 if isinstance(target, ForecastProbeTarget) else target
        identity = require_sha256(identity, field="target_sha256")
        try:
            return self._targets[identity]
        except KeyError as exc:
            raise KeyError("probe target is not registered") from exc

    def _total_work(self) -> float:
        return sum(receipt.work_cost for receipt in self._receipts)

    def _total_bytes(self) -> int:
        return sum(
            self._targets[receipt.target_sha256].expected_bytes
            for receipt in self._receipts
        )

    def _ordered_frontier(
        self,
        *,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> list[ForecastProbeTarget]:
        remaining_work = self.budgets.work - self._total_work()
        remaining_bytes = self.budgets.bytes - self._total_bytes()
        if maximum_work is not None:
            remaining_work = min(
                remaining_work,
                _finite(maximum_work, field="maximum_work", minimum=0.0),
            )
        if maximum_bytes is not None:
            remaining_bytes = min(
                remaining_bytes,
                _uint(maximum_bytes, field="maximum_bytes"),
            )
        candidates = [
            target
            for identity, target in self._targets.items()
            if identity not in self._settled
            and identity not in self._inflight
            and target.expected_work <= remaining_work
            and target.expected_bytes <= remaining_bytes
        ]
        estimates = {target.sha256: self.priority_breakdown(target) for target in candidates}
        candidates.sort(
            key=lambda target: (
                -estimates[target.sha256].priority,
                -estimates[target.sha256].value_density,
                target.sha256,
            )
        )
        return candidates

    def frontier(
        self,
        *,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> tuple[ForecastProbeTarget, ...]:
        with self._lock:
            return tuple(
                self._ordered_frontier(
                    maximum_work=maximum_work,
                    maximum_bytes=maximum_bytes,
                )
            )

    def next_job(
        self,
        *,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> ForecastProbeTarget | None:
        """Return the highest-value feasible job without executing or claiming it."""

        with self._lock:
            frontier = self._ordered_frontier(
                maximum_work=maximum_work,
                maximum_bytes=maximum_bytes,
            )
            return frontier[0] if frontier else None

    def _evidence_information(self, evidence: ProbeEvidence) -> float:
        return (
            self.config.coverage_weight * abs(evidence.coverage_gap)
            + self.config.joint_miss_weight * evidence.joint_miss
            + self.config.surprise_weight * evidence.o1_surprise
            + self.config.route_w1_weight * evidence.route_w1
            + self.config.learning_progress_weight
            * abs(evidence.learning_progress)
        )

    def settle(
        self,
        evidence: ProbeEvidence,
        *,
        committed_at: float | None = None,
    ) -> ProbeReceipt:
        """Commit realized evidence once; exact replay returns the first receipt."""

        if not isinstance(evidence, ProbeEvidence):
            raise TypeError("evidence must be ProbeEvidence")
        committed = time.time() if committed_at is None else committed_at
        committed = _finite(committed, field="committed_at", minimum=0.0)
        with self._lock:
            target = self._resolve_target(evidence.target_sha256)
            previous = self._settled.get(target.sha256)
            if previous is not None:
                if previous.evidence.sha256 == evidence.sha256:
                    return previous
                raise ProbeConflictError(
                    "probe target was already settled with different evidence"
                )
            if evidence.work_cost > target.expected_work:
                raise CartographyBudgetError(
                    "realized work exceeds the target's declared expected work"
                )
            if self._total_work() + evidence.work_cost > self.budgets.work:
                raise CartographyBudgetError("cartographer work budget would be exceeded")
            if self._total_bytes() + target.expected_bytes > self.budgets.bytes:
                raise CartographyBudgetError("cartographer byte budget would be exceeded")
            estimate = self.priority_breakdown(target)
            receipt = ProbeReceipt(
                sequence=len(self._receipts) + 1,
                target_sha256=target.sha256,
                evidence=evidence,
                previous_receipt_sha256=(
                    self._receipts[-1].sha256 if self._receipts else ZERO_SHA256
                ),
                priority_at_settlement=estimate.priority,
                information_gain=self._evidence_information(evidence),
                committed_at=committed,
            )
            event = {
                "receipt": receipt.to_record(),
                "receipt_sha256": receipt.sha256,
                "schema": EVENT_SCHEMA,
            }
            encoded = canonical_json_bytes(event)
            if len(encoded) > MAX_EVENT_BYTES:
                raise CartographyIntegrityError("probe event exceeds its byte bound")
            event_path = self.event_root / (
                f"{receipt.sequence:020d}-{receipt.sha256}.json"
            )
            _append_immutable(event_path, encoded)
            self._receipts.append(receipt)
            self._settled[target.sha256] = receipt
            self._write_checkpoint()
            return receipt

    def step(
        self,
        executor: ProbeExecutor,
        *,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> ProbeReceipt | None:
        """Execute and settle at most one feasible probe through ``executor``."""

        if not callable(executor):
            raise TypeError("executor must be callable")
        with self._lock:
            frontier = self._ordered_frontier(
                maximum_work=maximum_work,
                maximum_bytes=maximum_bytes,
            )
            if not frontier:
                return None
            target = frontier[0]
            self._inflight.add(target.sha256)
        try:
            evidence = executor(target)
            if not isinstance(evidence, ProbeEvidence):
                raise TypeError("executor must return ProbeEvidence")
            if evidence.target_sha256 != target.sha256:
                raise ProbeConflictError("executor returned evidence for another target")
            return self.settle(evidence)
        finally:
            with self._lock:
                self._inflight.discard(target.sha256)

    def run(
        self,
        executor: ProbeExecutor,
        *,
        max_jobs: int = 1,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> tuple[ProbeReceipt, ...]:
        """Run a caller-bounded number of serial probes."""

        jobs = _uint(max_jobs, field="max_jobs", positive=True, maximum=MAX_TARGETS)
        local_work = (
            None
            if maximum_work is None
            else _finite(maximum_work, field="maximum_work", minimum=0.0)
        )
        local_bytes = (
            None
            if maximum_bytes is None
            else _uint(maximum_bytes, field="maximum_bytes")
        )
        receipts: list[ProbeReceipt] = []
        consumed_work = 0.0
        consumed_bytes = 0
        for _ in range(jobs):
            available_work = (
                None if local_work is None else local_work - consumed_work
            )
            available_bytes = (
                None if local_bytes is None else local_bytes - consumed_bytes
            )
            receipt = self.step(
                executor,
                maximum_work=available_work,
                maximum_bytes=available_bytes,
            )
            if receipt is None:
                break
            receipts.append(receipt)
            consumed_work += receipt.work_cost
            consumed_bytes += self._targets[receipt.target_sha256].expected_bytes
        return tuple(receipts)

    def d_optimal_batch(
        self,
        feature_vectors: Mapping[
            ForecastProbeTarget | str, Sequence[float] | np.ndarray
        ],
        *,
        batch_size: int,
        ridge: float | None = None,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> tuple[ForecastProbeTarget, ...]:
        """Greedily acquire a deterministic diverse batch by log-det gain.

        The vectors are supplied by the caller and may contain only information
        available before settlement.  They are never persisted by this method.
        """

        size = _uint(
            batch_size,
            field="batch_size",
            positive=True,
            maximum=MAX_TARGETS,
        )
        ridge_value = self.config.d_optimal_ridge if ridge is None else ridge
        ridge_value = _positive_float(ridge_value, field="ridge")
        if not isinstance(feature_vectors, Mapping) or not feature_vectors:
            raise ValueError("feature_vectors must be a non-empty mapping")
        with self._lock:
            vectors: dict[str, np.ndarray] = {}
            dimension: int | None = None
            for key, raw in feature_vectors.items():
                target = self._resolve_target(key)
                if target.sha256 in vectors:
                    raise ValueError("feature_vectors contains a duplicate target")
                array = np.asarray(raw, dtype=np.float64)
                if array.ndim != 1 or array.size == 0:
                    raise ValueError("feature vectors must be non-empty and one-dimensional")
                if array.size > self.config.max_feature_dimension:
                    raise ValueError("feature vector dimension exceeds its bound")
                if not np.all(np.isfinite(array)):
                    raise ValueError("feature vectors must be finite")
                if np.any(np.abs(array) > self.config.max_feature_absolute):
                    raise ValueError("feature vector magnitude exceeds its bound")
                if dimension is None:
                    dimension = int(array.size)
                elif array.size != dimension:
                    raise ValueError("feature vectors must share one dimension")
                if target.sha256 not in self._settled and target.sha256 not in self._inflight:
                    vectors[target.sha256] = np.ascontiguousarray(array)
            if not vectors:
                return ()
            assert dimension is not None
            global_work = self.budgets.work - self._total_work()
            global_bytes = self.budgets.bytes - self._total_bytes()
            if maximum_work is not None:
                global_work = min(
                    global_work,
                    _finite(maximum_work, field="maximum_work", minimum=0.0),
                )
            if maximum_bytes is not None:
                global_bytes = min(
                    global_bytes,
                    _uint(maximum_bytes, field="maximum_bytes"),
                )
            information = np.eye(dimension, dtype=np.float64) * ridge_value
            selected: list[ForecastProbeTarget] = []
            selected_work = 0.0
            selected_bytes = 0
            remaining = dict(vectors)
            estimates = {
                identity: self.priority_breakdown(identity)
                for identity in remaining
            }
            while remaining and len(selected) < size:
                choices: list[
                    tuple[float, float, float, str, ForecastProbeTarget]
                ] = []
                for identity in sorted(remaining):
                    target = self._targets[identity]
                    if (
                        selected_work + target.expected_work > global_work
                        or selected_bytes + target.expected_bytes > global_bytes
                    ):
                        continue
                    vector = remaining[identity]
                    try:
                        solved = np.linalg.solve(information, vector)
                    except np.linalg.LinAlgError as exc:  # Positive ridge is invariant.
                        raise CartographyError("D-optimal information matrix failed") from exc
                    quadratic = max(0.0, float(vector @ solved))
                    gain = math.log1p(quadratic)
                    estimate = estimates[identity]
                    choices.append(
                        (
                            gain,
                            estimate.priority,
                            estimate.value_density,
                            identity,
                            target,
                        )
                    )
                if not choices:
                    break
                # Identity is deliberately ascending for a stable final tie.
                choices.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
                _, _, _, identity, target = choices[0]
                vector = remaining.pop(identity)
                information += np.outer(vector, vector)
                selected.append(target)
                selected_work += target.expected_work
                selected_bytes += target.expected_bytes
            return tuple(selected)

    def acquire_batch(
        self,
        feature_vectors: Mapping[
            ForecastProbeTarget | str, Sequence[float] | np.ndarray
        ],
        *,
        batch_size: int,
        ridge: float | None = None,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> tuple[ForecastProbeTarget, ...]:
        return self.d_optimal_batch(
            feature_vectors,
            batch_size=batch_size,
            ridge=ridge,
            maximum_work=maximum_work,
            maximum_bytes=maximum_bytes,
        )

    def metrics(self) -> dict[str, object]:
        """Return an exact snapshot of frontier, debt, work, and information."""

        with self._lock:
            total_work = self._total_work()
            total_bytes = self._total_bytes()
            observed = sum(receipt.observed_values for receipt in self._receipts)
            debt_sum = sum(
                abs(receipt.coverage_gap) * receipt.observed_values
                for receipt in self._receipts
            )
            pending = len(self._targets) - len(self._settled)
            frontier = len(self._ordered_frontier())
            return {
                "completed": len(self._settled),
                "coverage_debt": 0.0 if observed == 0 else debt_sum / observed,
                "coverage_debt_sum": debt_sum,
                "frontier": frontier,
                "head_receipt_sha256": (
                    self._receipts[-1].sha256 if self._receipts else ZERO_SHA256
                ),
                "information_gain": sum(
                    receipt.information_gain for receipt in self._receipts
                ),
                "observed_values": observed,
                "pending": pending,
                "remaining_bytes": self.budgets.bytes - total_bytes,
                "remaining_work": self.budgets.work - total_work,
                "sequence": len(self._receipts),
                "targets": len(self._targets),
                "total_bytes": total_bytes,
                "total_work": total_work,
            }

    def _background_worker(self) -> None:
        while True:
            item = self._background_queue.get()
            if item is None:
                self._background_queue.task_done()
                return
            executor, max_jobs, maximum_work, maximum_bytes, handle = item
            try:
                receipts = self.run(
                    executor,
                    max_jobs=max_jobs,
                    maximum_work=maximum_work,
                    maximum_bytes=maximum_bytes,
                )
            except BaseException as exc:  # Forward the exact executor failure.
                handle._finish(None, exc)
            else:
                handle._finish(receipts, None)
            finally:
                self._background_queue.task_done()

    def submit_background(
        self,
        executor: ProbeExecutor,
        *,
        max_jobs: int = 1,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> BackgroundRun:
        """Queue one bounded run on the cartographer's single serial worker."""

        if not callable(executor):
            raise TypeError("executor must be callable")
        jobs = _uint(max_jobs, field="max_jobs", positive=True, maximum=MAX_TARGETS)
        if maximum_work is not None:
            maximum_work = _finite(
                maximum_work,
                field="maximum_work",
                minimum=0.0,
            )
        if maximum_bytes is not None:
            maximum_bytes = _uint(maximum_bytes, field="maximum_bytes")
        with self._lock:
            if self._background_closed:
                raise CartographyError("background runner is closed")
            if self._background_thread is None:
                self._background_thread = threading.Thread(
                    target=self._background_worker,
                    name="o1fm-forecast-cartographer",
                    daemon=True,
                )
                self._background_thread.start()
            handle = BackgroundRun()
            self._background_queue.put(
                (executor, jobs, maximum_work, maximum_bytes, handle)
            )
            return handle

    def background_run(
        self,
        executor: ProbeExecutor,
        *,
        max_jobs: int = 1,
        maximum_work: float | None = None,
        maximum_bytes: int | None = None,
    ) -> BackgroundRun:
        return self.submit_background(
            executor,
            max_jobs=max_jobs,
            maximum_work=maximum_work,
            maximum_bytes=maximum_bytes,
        )

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._background_closed:
                thread = self._background_thread
            else:
                self._background_closed = True
                thread = self._background_thread
                if thread is not None:
                    self._background_queue.put(None)
        if wait and thread is not None:
            thread.join()

    def __enter__(self) -> ForecastCartographer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "ZERO_SHA256",
    "BackgroundRun",
    "CartographerBudgets",
    "CartographerConfig",
    "CartographyBudgetError",
    "CartographyError",
    "CartographyIntegrityError",
    "ForecastCartographer",
    "ForecastProbeTarget",
    "ProbeConflictError",
    "ProbeEvidence",
    "ProbeExecutor",
    "ProbePriority",
    "ProbeReceipt",
]
