"""Persistent continuous forecast-error state with online regime detection."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any
import unicodedata

import numpy as np

from .identity import canonical_json_bytes, require_sha256, sha256_json

try:  # POSIX is the production target; the process-local lock remains portable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


O1_STATE_SCHEMA = "o1fm.o1-error-state/v1"
O1_EVENT_SCHEMA = "o1fm.o1-error-event/v1"
O1_CONFIG_SCHEMA = "o1fm.o1-error-config/v1"
ZERO_SHA256 = "0" * 64

MAX_EVENT_BYTES = 256 * 1024
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_VALUES_PER_OBSERVATION = 1_000_000
MAX_EVENTS = 1_000_000
MAX_COUNTER = (1 << 63) - 1

_EVENT_NAME = re.compile(r"([0-9]{20})-([0-9a-f]{64})\.json\Z")
_STREAM_ID_BYTES = 1024
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


class O1Error(RuntimeError):
    """The O1 stream input or persistent state is invalid."""


class O1IntegrityError(O1Error):
    """An O1 state or event failed its canonical integrity contract."""


class O1ConflictError(O1Error):
    """An observation identity was replayed with different evidence."""


def _finite(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{field} must be finite and non-negative")
    return 0.0 if result == 0.0 else result


def _probability(value: object, *, field: str, open_interval: bool = False) -> float:
    result = _finite(value, field=field, nonnegative=True)
    valid = 0.0 < result < 1.0 if open_interval else 0.0 <= result <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{field} must lie in {interval}")
    return result


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not int(positive) <= value <= MAX_COUNTER
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return value


def _checked_add(left: int, right: int, *, field: str) -> int:
    result = left + right
    return _uint(result, field=field)


def _hex(value: float) -> str:
    return _finite(value, field="float").hex()


def _from_hex(value: object, *, field: str, nonnegative: bool = False) -> float:
    if not isinstance(value, str) or len(value) > 32:
        raise O1IntegrityError(f"{field} is not a canonical float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise O1IntegrityError(f"{field} is not a canonical float") from exc
    result = _finite(result, field=field, nonnegative=nonnegative)
    if result.hex() != value:
        raise O1IntegrityError(f"{field} is not canonical hexadecimal float")
    return result


def _stream_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > _STREAM_ID_BYTES
    ):
        raise ValueError("stream_id must be canonical bounded text")
    return value


@dataclass(frozen=True, slots=True)
class O1Config:
    """Pinned update policy for one continuous forecast-error stream."""

    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    ew_decay: float = 0.95
    cusum_decay: float = 0.90
    cusum_drift: float = 0.50
    cusum_threshold: float = 8.0
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.quantiles, (str, bytes)):
            raise TypeError("quantiles must be a numeric sequence")
        values = tuple(
            _probability(value, field="quantile", open_interval=True)
            for value in self.quantiles
        )
        if not values or values != tuple(sorted(set(values))):
            raise ValueError("quantiles must be sorted and unique")
        object.__setattr__(self, "quantiles", values)
        object.__setattr__(
            self,
            "ew_decay",
            _probability(self.ew_decay, field="ew_decay", open_interval=True),
        )
        object.__setattr__(
            self,
            "cusum_decay",
            _probability(self.cusum_decay, field="cusum_decay"),
        )
        object.__setattr__(
            self,
            "cusum_drift",
            _finite(self.cusum_drift, field="cusum_drift", nonnegative=True),
        )
        threshold = _finite(
            self.cusum_threshold,
            field="cusum_threshold",
            nonnegative=True,
        )
        if threshold == 0.0:
            raise ValueError("cusum_threshold must be positive")
        object.__setattr__(self, "cusum_threshold", threshold)
        epsilon = _finite(self.epsilon, field="epsilon", nonnegative=True)
        if epsilon == 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "epsilon", epsilon)

    def to_record(self) -> dict[str, object]:
        return {
            "cusum_decay": self.cusum_decay.hex(),
            "cusum_drift": self.cusum_drift.hex(),
            "cusum_threshold": self.cusum_threshold.hex(),
            "epsilon": self.epsilon.hex(),
            "ew_decay": self.ew_decay.hex(),
            "quantiles": [value.hex() for value in self.quantiles],
            "schema": O1_CONFIG_SCHEMA,
        }

    @classmethod
    def from_record(cls, value: object) -> O1Config:
        fields = {
            "cusum_decay",
            "cusum_drift",
            "cusum_threshold",
            "epsilon",
            "ew_decay",
            "quantiles",
            "schema",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise O1IntegrityError("O1 config fields are invalid")
        if value.get("schema") != O1_CONFIG_SCHEMA or not isinstance(
            value.get("quantiles"),
            list,
        ):
            raise O1IntegrityError("O1 config schema is invalid")
        try:
            result = cls(
                quantiles=tuple(
                    _from_hex(item, field="quantile")
                    for item in value["quantiles"]
                ),
                ew_decay=_from_hex(value["ew_decay"], field="ew_decay"),
                cusum_decay=_from_hex(
                    value["cusum_decay"], field="cusum_decay"
                ),
                cusum_drift=_from_hex(
                    value["cusum_drift"],
                    field="cusum_drift",
                    nonnegative=True,
                ),
                cusum_threshold=_from_hex(
                    value["cusum_threshold"],
                    field="cusum_threshold",
                    nonnegative=True,
                ),
                epsilon=_from_hex(
                    value["epsilon"], field="epsilon", nonnegative=True
                ),
            )
        except (TypeError, ValueError) as exc:
            raise O1IntegrityError("O1 config values are invalid") from exc
        if result.to_record() != dict(value):
            raise O1IntegrityError("O1 config is not canonical")
        return result

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_record())


@dataclass(frozen=True, slots=True)
class O1State:
    """Complete aggregate state after one committed event revision."""

    identity_sha256: str
    sequence: int
    head_event_sha256: str
    observations: int
    value_count: int
    absolute_error_sum: float
    signed_error_sum: float
    pinball_sums: tuple[float, ...]
    pinball_counts: tuple[int, ...]
    interval_hits: int
    interval_count: int
    interval_width_sum: float
    ew_normalized_error: float
    ew_absolute_error: float
    surprise: float
    learning_progress: float
    cusum: float
    regime_generation: int
    regime_changes: int

    def __post_init__(self) -> None:
        require_sha256(self.identity_sha256, field="identity_sha256")
        require_sha256(self.head_event_sha256, field="head_event_sha256")
        for name in (
            "sequence",
            "observations",
            "value_count",
            "interval_hits",
            "interval_count",
            "regime_generation",
            "regime_changes",
        ):
            _uint(getattr(self, name), field=name)
        if self.sequence != self.observations:
            raise ValueError("O1 sequence must equal committed observations")
        if self.interval_hits > self.interval_count:
            raise ValueError("interval hits exceed interval observations")
        for name in (
            "absolute_error_sum",
            "interval_width_sum",
            "ew_normalized_error",
            "ew_absolute_error",
            "surprise",
            "learning_progress",
            "cusum",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), field=name, nonnegative=True),
            )
        object.__setattr__(
            self,
            "signed_error_sum",
            _finite(self.signed_error_sum, field="signed_error_sum"),
        )
        pinball = tuple(
            _finite(value, field="pinball_sum", nonnegative=True)
            for value in self.pinball_sums
        )
        counts = tuple(
            _uint(value, field="pinball_count") for value in self.pinball_counts
        )
        if len(pinball) != len(counts):
            raise ValueError("pinball sum/count dimensions differ")
        object.__setattr__(self, "pinball_sums", pinball)
        object.__setattr__(self, "pinball_counts", counts)

    @classmethod
    def initial(cls, identity_sha256: str, quantile_count: int) -> O1State:
        return cls(
            identity_sha256=identity_sha256,
            sequence=0,
            head_event_sha256=ZERO_SHA256,
            observations=0,
            value_count=0,
            absolute_error_sum=0.0,
            signed_error_sum=0.0,
            pinball_sums=(0.0,) * quantile_count,
            pinball_counts=(0,) * quantile_count,
            interval_hits=0,
            interval_count=0,
            interval_width_sum=0.0,
            ew_normalized_error=0.0,
            ew_absolute_error=0.0,
            surprise=0.0,
            learning_progress=0.0,
            cusum=0.0,
            regime_generation=0,
            regime_changes=0,
        )

    @property
    def mae(self) -> float:
        return self.absolute_error_sum / max(1, self.value_count)

    @property
    def bias(self) -> float:
        """Mean ``actual - point_forecast``; positive means underforecasting."""

        return self.signed_error_sum / max(1, self.value_count)

    @property
    def interval_coverage(self) -> float | None:
        return (
            None
            if self.interval_count == 0
            else self.interval_hits / self.interval_count
        )

    @property
    def mean_interval_width(self) -> float | None:
        return (
            None
            if self.interval_count == 0
            else self.interval_width_sum / self.interval_count
        )

    def core_record(self) -> dict[str, object]:
        return {
            "absolute_error_sum": self.absolute_error_sum.hex(),
            "cusum": self.cusum.hex(),
            "ew_absolute_error": self.ew_absolute_error.hex(),
            "ew_normalized_error": self.ew_normalized_error.hex(),
            "identity_sha256": self.identity_sha256,
            "interval_count": self.interval_count,
            "interval_hits": self.interval_hits,
            "interval_width_sum": self.interval_width_sum.hex(),
            "learning_progress": self.learning_progress.hex(),
            "observations": self.observations,
            "pinball_counts": list(self.pinball_counts),
            "pinball_sums": [value.hex() for value in self.pinball_sums],
            "regime_changes": self.regime_changes,
            "regime_generation": self.regime_generation,
            "sequence": self.sequence,
            "signed_error_sum": self.signed_error_sum.hex(),
            "surprise": self.surprise.hex(),
            "value_count": self.value_count,
        }

    def to_record(self) -> dict[str, object]:
        return {**self.core_record(), "head_event_sha256": self.head_event_sha256}

    @classmethod
    def from_record(
        cls,
        value: object,
        *,
        head_event_sha256: str | None = None,
    ) -> O1State:
        core_fields = {
            "absolute_error_sum",
            "cusum",
            "ew_absolute_error",
            "ew_normalized_error",
            "identity_sha256",
            "interval_count",
            "interval_hits",
            "interval_width_sum",
            "learning_progress",
            "observations",
            "pinball_counts",
            "pinball_sums",
            "regime_changes",
            "regime_generation",
            "sequence",
            "signed_error_sum",
            "surprise",
            "value_count",
        }
        if not isinstance(value, Mapping) or set(value) not in (
            core_fields,
            core_fields | {"head_event_sha256"},
        ):
            raise O1IntegrityError("O1 state fields are invalid")
        if head_event_sha256 is None:
            head_event_sha256 = require_sha256(
                value.get("head_event_sha256"), field="head_event_sha256"
            )
        elif "head_event_sha256" in value:
            raise O1IntegrityError("event state must not contain its circular head")
        pinball_sums = value.get("pinball_sums")
        pinball_counts = value.get("pinball_counts")
        if not isinstance(pinball_sums, list) or not isinstance(
            pinball_counts, list
        ):
            raise O1IntegrityError("O1 pinball fields are invalid")
        try:
            return cls(
                identity_sha256=value["identity_sha256"],
                sequence=value["sequence"],
                head_event_sha256=head_event_sha256,
                observations=value["observations"],
                value_count=value["value_count"],
                absolute_error_sum=_from_hex(
                    value["absolute_error_sum"],
                    field="absolute_error_sum",
                    nonnegative=True,
                ),
                signed_error_sum=_from_hex(
                    value["signed_error_sum"], field="signed_error_sum"
                ),
                pinball_sums=tuple(
                    _from_hex(item, field="pinball_sum", nonnegative=True)
                    for item in pinball_sums
                ),
                pinball_counts=tuple(pinball_counts),
                interval_hits=value["interval_hits"],
                interval_count=value["interval_count"],
                interval_width_sum=_from_hex(
                    value["interval_width_sum"],
                    field="interval_width_sum",
                    nonnegative=True,
                ),
                ew_normalized_error=_from_hex(
                    value["ew_normalized_error"],
                    field="ew_normalized_error",
                    nonnegative=True,
                ),
                ew_absolute_error=_from_hex(
                    value["ew_absolute_error"],
                    field="ew_absolute_error",
                    nonnegative=True,
                ),
                surprise=_from_hex(
                    value["surprise"], field="surprise", nonnegative=True
                ),
                learning_progress=_from_hex(
                    value["learning_progress"],
                    field="learning_progress",
                    nonnegative=True,
                ),
                cusum=_from_hex(value["cusum"], field="cusum", nonnegative=True),
                regime_generation=value["regime_generation"],
                regime_changes=value["regime_changes"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise O1IntegrityError("O1 state values are invalid") from exc


@dataclass(frozen=True, slots=True)
class O1Update:
    observation_sha256: str
    sequence: int
    duplicate: bool
    regime_changed: bool
    regime_generation: int
    batch_mae: float
    batch_bias: float
    normalized_error: float
    surprise: float
    learning_progress: float
    cusum: float
    interval_coverage: float | None
    pinball: Mapping[str, float]

    def __post_init__(self) -> None:
        require_sha256(self.observation_sha256, field="observation_sha256")
        _uint(self.sequence, field="sequence")
        if not isinstance(self.duplicate, bool) or not isinstance(
            self.regime_changed, bool
        ):
            raise TypeError("duplicate/regime_changed must be boolean")
        _uint(self.regime_generation, field="regime_generation")
        for name in (
            "batch_mae",
            "normalized_error",
            "surprise",
            "learning_progress",
            "cusum",
        ):
            _finite(getattr(self, name), field=name, nonnegative=True)
        _finite(self.batch_bias, field="batch_bias")
        if self.interval_coverage is not None:
            _probability(self.interval_coverage, field="interval_coverage")
        if any(
            not isinstance(key, str)
            or _finite(value, field="pinball", nonnegative=True) < 0.0
            for key, value in self.pinball.items()
        ):
            raise ValueError("pinball update is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "batch_bias": self.batch_bias.hex(),
            "batch_mae": self.batch_mae.hex(),
            "cusum": self.cusum.hex(),
            "interval_coverage": (
                None
                if self.interval_coverage is None
                else self.interval_coverage.hex()
            ),
            "learning_progress": self.learning_progress.hex(),
            "normalized_error": self.normalized_error.hex(),
            "observation_sha256": self.observation_sha256,
            "pinball": {
                key: value.hex() for key, value in sorted(self.pinball.items())
            },
            "regime_changed": self.regime_changed,
            "regime_generation": self.regime_generation,
            "sequence": self.sequence,
            "surprise": self.surprise.hex(),
        }

    @classmethod
    def from_record(cls, value: object, *, duplicate: bool) -> O1Update:
        fields = {
            "batch_bias",
            "batch_mae",
            "cusum",
            "interval_coverage",
            "learning_progress",
            "normalized_error",
            "observation_sha256",
            "pinball",
            "regime_changed",
            "regime_generation",
            "sequence",
            "surprise",
        }
        if not isinstance(value, Mapping) or set(value) != fields or not isinstance(
            value.get("pinball"), Mapping
        ):
            raise O1IntegrityError("O1 update fields are invalid")
        coverage = value["interval_coverage"]
        return cls(
            observation_sha256=value["observation_sha256"],
            sequence=value["sequence"],
            duplicate=duplicate,
            regime_changed=value["regime_changed"],
            regime_generation=value["regime_generation"],
            batch_mae=_from_hex(value["batch_mae"], field="batch_mae"),
            batch_bias=_from_hex(value["batch_bias"], field="batch_bias"),
            normalized_error=_from_hex(
                value["normalized_error"], field="normalized_error"
            ),
            surprise=_from_hex(value["surprise"], field="surprise"),
            learning_progress=_from_hex(
                value["learning_progress"], field="learning_progress"
            ),
            cusum=_from_hex(value["cusum"], field="cusum"),
            interval_coverage=(
                None
                if coverage is None
                else _from_hex(coverage, field="interval_coverage")
            ),
            pinball={
                str(key): _from_hex(child, field="pinball")
                for key, child in value["pinball"].items()
            },
        )


def _thread_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.absolute()))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    lock_path = path.parent / f".{path.name}.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT
        | os.O_RDWR
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise O1IntegrityError("O1 lock must be a regular file")
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _stable_read(path: Path, *, maximum: int, label: str) -> bytes:
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise O1IntegrityError(f"cannot open {label}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            or not 0 < opened.st_size <= maximum
        ):
            raise O1IntegrityError(f"{label} is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                raise O1IntegrityError(f"{label} was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise O1IntegrityError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        if (
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_dev,
            opened.st_ino,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_dev,
            after.st_ino,
        ):
            raise O1IntegrityError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_document(raw: bytes, *, label: str) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, child in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = child
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise O1IntegrityError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise O1IntegrityError(f"{label} is not canonical JSON")
    return value


class O1ErrorStream:
    """Settlement-only online forecast-error stream.

    ``actual`` is accepted only here, after a forecast exists.  No method on
    this class constructs model inputs, which keeps realized futures out of the
    prediction path by construction.
    """

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        *,
        stream_id: str = "default",
        config: O1Config | None = None,
    ) -> None:
        self.state_path = Path(state_path).expanduser().absolute()
        if not self.state_path.name:
            raise ValueError("state_path must name a file")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.state_path.parent.lstat()
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise O1IntegrityError("O1 state parent must be a real directory")
        self.events_path = self.state_path.parent / f"{self.state_path.name}.events"
        self.events_path.mkdir(exist_ok=True, mode=0o700)
        events_metadata = self.events_path.lstat()
        if stat.S_ISLNK(events_metadata.st_mode) or not stat.S_ISDIR(
            events_metadata.st_mode
        ):
            raise O1IntegrityError("O1 events path must be a real directory")
        self.stream_id = _stream_id(stream_id)
        self.config = O1Config() if config is None else config
        if not isinstance(self.config, O1Config):
            raise TypeError("config must be O1Config")
        self.identity_sha256 = sha256_json(
            {
                "config_sha256": self.config.sha256,
                "schema": "o1fm.o1-error-stream-identity/v1",
                "stream_id": self.stream_id,
            }
        )
        self._lock = _thread_lock(self.state_path)
        self._state = O1State.initial(
            self.identity_sha256,
            len(self.config.quantiles),
        )
        self._observations: dict[str, tuple[str, O1Update]] = {}
        with self._lock, _state_lock(self.state_path):
            self._load_locked()

    @property
    def state(self) -> O1State:
        with self._lock:
            return self._state

    def _event_paths(self) -> tuple[Path, ...]:
        paths = tuple(sorted(self.events_path.iterdir(), key=lambda item: item.name))
        if len(paths) > MAX_EVENTS:
            raise O1IntegrityError("O1 event inventory exceeds its bound")
        if any(
            not stat.S_ISREG(path.lstat().st_mode)
            or _EVENT_NAME.fullmatch(path.name) is None
            for path in paths
        ):
            raise O1IntegrityError("O1 event inventory is invalid")
        return paths

    def _load_events_locked(self) -> O1State:
        state = O1State.initial(self.identity_sha256, len(self.config.quantiles))
        observations: dict[str, tuple[str, O1Update]] = {}
        previous = ZERO_SHA256
        for expected_sequence, path in enumerate(self._event_paths(), start=1):
            match = _EVENT_NAME.fullmatch(path.name)
            assert match is not None
            if int(match.group(1)) != expected_sequence:
                raise O1IntegrityError("O1 event sequence is not contiguous")
            raw = _stable_read(path, maximum=MAX_EVENT_BYTES, label="O1 event")
            document = _strict_document(raw, label="O1 event")
            body = document.get("body")
            if (
                set(document) != {"body", "schema", "sha256"}
                or document.get("schema") != O1_EVENT_SCHEMA
                or not isinstance(body, Mapping)
                or document.get("sha256") != sha256_json(body)
                or document.get("sha256") != match.group(2)
                or set(body)
                != {
                    "identity_sha256",
                    "observation_id",
                    "observation_sha256",
                    "previous_event_sha256",
                    "sequence",
                    "state",
                    "update",
                }
                or body.get("sequence") != expected_sequence
                or body.get("previous_event_sha256") != previous
                or body.get("identity_sha256") != self.identity_sha256
                or not isinstance(body.get("state"), Mapping)
                or not isinstance(body.get("update"), Mapping)
            ):
                raise O1IntegrityError("O1 event envelope is invalid")
            observation_id = _stream_id(body.get("observation_id"))
            observation_sha256 = require_sha256(
                body.get("observation_sha256"), field="observation_sha256"
            )
            if observation_id in observations:
                raise O1IntegrityError("O1 event repeats an observation identity")
            state = O1State.from_record(
                body["state"],
                head_event_sha256=document["sha256"],
            )
            if (
                state.sequence != expected_sequence
                or state.identity_sha256 != self.identity_sha256
                or len(state.pinball_sums) != len(self.config.quantiles)
            ):
                raise O1IntegrityError("O1 event state is incompatible")
            update = O1Update.from_record(body["update"], duplicate=False)
            if (
                update.sequence != expected_sequence
                or update.observation_sha256 != observation_sha256
                or update.regime_generation != state.regime_generation
            ):
                raise O1IntegrityError("O1 event update disagrees with state")
            observations[observation_id] = (observation_sha256, update)
            previous = document["sha256"]
        self._observations = observations
        return state

    def _load_locked(self) -> None:
        event_state = self._load_events_locked()
        try:
            raw = _stable_read(
                self.state_path,
                maximum=MAX_STATE_BYTES,
                label="O1 state",
            )
        except O1IntegrityError as exc:
            if not self.state_path.exists():
                self._state = event_state
                return
            raise exc
        document = _strict_document(raw, label="O1 state")
        body = document.get("body")
        if (
            set(document) != {"body", "schema", "sha256"}
            or document.get("schema") != O1_STATE_SCHEMA
            or not isinstance(body, Mapping)
            or document.get("sha256") != sha256_json(body)
        ):
            raise O1IntegrityError("O1 state envelope is invalid")
        persisted = O1State.from_record(body)
        if (
            persisted.identity_sha256 != self.identity_sha256
            or len(persisted.pinball_sums) != len(self.config.quantiles)
            or persisted.sequence > event_state.sequence
            or (
                persisted.sequence == event_state.sequence
                and persisted != event_state
            )
        ):
            raise O1IntegrityError("O1 state differs from its event history")
        self._state = event_state

    def _write_state_locked(self, state: O1State) -> None:
        body = state.to_record()
        data = canonical_json_bytes(
            {"body": body, "schema": O1_STATE_SCHEMA, "sha256": sha256_json(body)}
        )
        if len(data) > MAX_STATE_BYTES:
            raise O1IntegrityError("O1 state exceeds its byte bound")
        _atomic_write(self.state_path, data)

    def _publish_event_locked(
        self,
        *,
        observation_id: str,
        observation_sha256: str,
        state: O1State,
        update: O1Update,
    ) -> O1State:
        body = {
            "identity_sha256": self.identity_sha256,
            "observation_id": observation_id,
            "observation_sha256": observation_sha256,
            "previous_event_sha256": self._state.head_event_sha256,
            "sequence": state.sequence,
            "state": state.core_record(),
            "update": update.to_record(),
        }
        event_sha256 = sha256_json(body)
        data = canonical_json_bytes(
            {"body": body, "schema": O1_EVENT_SCHEMA, "sha256": event_sha256}
        )
        if len(data) > MAX_EVENT_BYTES:
            raise O1IntegrityError("O1 event exceeds its byte bound")
        destination = self.events_path / f"{state.sequence:020d}-{event_sha256}.json"
        temporary = self.events_path / f".{destination.name}.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            _write_all(descriptor, data)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, destination, follow_symlinks=False)
            _fsync_directory(self.events_path)
        except FileExistsError as exc:
            raise O1ConflictError("O1 event sequence already exists") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return O1State.from_record(state.core_record(), head_event_sha256=event_sha256)

    def observe(
        self,
        actual: object,
        point_forecast: object,
        *,
        quantile_forecasts: Mapping[float, object] | None = None,
        lower: object | None = None,
        upper: object | None = None,
        scale: object | None = None,
        mask: object | None = None,
        observation_id: str | None = None,
    ) -> O1Update:
        """Settle one realized forecast and return its committed update."""

        actual_values = np.asarray(actual)
        point_values = np.asarray(point_forecast)
        try:
            actual_array = np.array(actual, dtype=np.float64, order="C", copy=True)
            point_array = np.array(
                point_forecast, dtype=np.float64, order="C", copy=True
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("actual and point_forecast must be numeric") from exc
        if (
            actual_values.ndim < 1
            or point_values.ndim != actual_values.ndim
            or actual_array.shape != point_array.shape
            or not 1 <= actual_array.size <= MAX_VALUES_PER_OBSERVATION
        ):
            raise ValueError("actual and point_forecast must have equal bounded shape")
        if mask is None:
            observed = np.ones(actual_array.shape, dtype=np.bool_)
        else:
            observed = np.asarray(mask)
            if observed.dtype != np.dtype(np.bool_) or observed.shape != actual_array.shape:
                raise ValueError("mask must be boolean with the forecast shape")
        if not bool(observed.any()):
            raise ValueError("at least one realized value must be observed")
        if bool(
            (~np.isfinite(actual_array[observed])).any()
            or (~np.isfinite(point_array[observed])).any()
        ):
            raise ValueError("observed actual/forecast values must be finite")
        actual_array[~observed] = 0.0
        point_array[~observed] = 0.0

        quantile_arrays: dict[float, np.ndarray[Any, Any]] = {}
        if quantile_forecasts is not None:
            if not isinstance(quantile_forecasts, Mapping):
                raise TypeError("quantile_forecasts must be a mapping")
            for raw_level, raw_forecast in quantile_forecasts.items():
                level = _probability(
                    raw_level, field="quantile", open_interval=True
                )
                if level not in self.config.quantiles:
                    raise ValueError("quantile is outside the configured inventory")
                try:
                    forecast = np.array(
                        raw_forecast,
                        dtype=np.float64,
                        order="C",
                        copy=True,
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("quantile forecast must be numeric") from exc
                if forecast.shape != actual_array.shape or not bool(
                    np.isfinite(forecast[observed]).all()
                ):
                    raise ValueError("quantile forecast shape/data is invalid")
                forecast[~observed] = 0.0
                quantile_arrays[level] = forecast
            ordered = sorted(quantile_arrays)
            for left_level, right_level in zip(ordered, ordered[1:], strict=False):
                if bool(
                    (
                        quantile_arrays[left_level][observed]
                        > quantile_arrays[right_level][observed]
                    ).any()
                ):
                    raise ValueError("quantile forecasts must be non-decreasing")

        def interval_array(value: object, field: str) -> np.ndarray[Any, Any]:
            try:
                result = np.array(value, dtype=np.float64, order="C", copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{field} must be numeric") from exc
            if result.shape != actual_array.shape or not bool(
                np.isfinite(result[observed]).all()
            ):
                raise ValueError(f"{field} shape/data is invalid")
            result[~observed] = 0.0
            return result

        if (lower is None) != (upper is None):
            raise ValueError("lower and upper must be supplied together")
        lower_array: np.ndarray[Any, Any] | None = None
        upper_array: np.ndarray[Any, Any] | None = None
        if lower is not None:
            lower_array = interval_array(lower, "lower")
            upper_array = interval_array(upper, "upper")
        elif len(quantile_arrays) >= 2:
            ordered = sorted(quantile_arrays)
            lower_array = quantile_arrays[ordered[0]]
            upper_array = quantile_arrays[ordered[-1]]
        if lower_array is not None and upper_array is not None and bool(
            (lower_array[observed] > upper_array[observed]).any()
        ):
            raise ValueError("lower interval exceeds upper interval")

        if scale is None:
            scale_array = (
                np.ones(actual_array.shape, dtype=np.float64)
                if lower_array is None or upper_array is None
                else np.maximum(
                    (upper_array - lower_array) * 0.5,
                    self.config.epsilon,
                )
            )
        else:
            raw_scale = np.asarray(scale)
            if raw_scale.ndim == 0:
                scale_array = np.full(
                    actual_array.shape,
                    _finite(float(raw_scale), field="scale", nonnegative=True),
                )
            else:
                scale_array = interval_array(scale, "scale")
            if bool((scale_array[observed] <= 0.0).any()):
                raise ValueError("scale must be positive at observed positions")

        residual = actual_array[observed] - point_array[observed]
        absolute = np.abs(residual)
        count = int(residual.size)
        batch_mae = float(absolute.mean())
        batch_bias = float(residual.mean())
        normalized_error = float(
            np.mean(absolute / (scale_array[observed] + self.config.epsilon))
        )
        pinball: dict[float, float] = {}
        for level, forecast in quantile_arrays.items():
            error = actual_array[observed] - forecast[observed]
            pinball[level] = float(
                np.maximum(level * error, (level - 1.0) * error).mean()
            )
        interval_hits = interval_count = 0
        interval_width_sum = 0.0
        coverage: float | None = None
        if lower_array is not None and upper_array is not None:
            hits = (actual_array[observed] >= lower_array[observed]) & (
                actual_array[observed] <= upper_array[observed]
            )
            interval_hits = int(hits.sum())
            interval_count = count
            interval_width_sum = float(
                (upper_array[observed] - lower_array[observed]).sum()
            )
            coverage = interval_hits / interval_count

        observation_record = {
            "actual": actual_array.tolist(),
            "lower": None if lower_array is None else lower_array.tolist(),
            "mask": observed.tolist(),
            "point": point_array.tolist(),
            "quantiles": {
                level.hex(): forecast.tolist()
                for level, forecast in sorted(quantile_arrays.items())
            },
            "scale": scale_array.tolist(),
            "schema": "o1fm.o1-settlement/v1",
            "stream_identity_sha256": self.identity_sha256,
            "upper": None if upper_array is None else upper_array.tolist(),
        }
        observation_sha256 = sha256_json(observation_record)
        selected_id = (
            f"sha256:{observation_sha256}"
            if observation_id is None
            else _stream_id(observation_id)
        )

        with self._lock, _state_lock(self.state_path):
            self._load_locked()
            prior = self._observations.get(selected_id)
            if prior is not None:
                if prior[0] != observation_sha256:
                    raise O1ConflictError(
                        "observation_id is already bound to different evidence"
                    )
                stored = prior[1]
                return O1Update.from_record(stored.to_record(), duplicate=True)
            state = self._state
            previous_ew = (
                normalized_error
                if state.observations == 0
                else state.ew_normalized_error
            )
            surprise = max(0.0, normalized_error - previous_ew)
            learning_progress = max(0.0, previous_ew - normalized_error)
            ew_normalized = (
                normalized_error
                if state.observations == 0
                else self.config.ew_decay * state.ew_normalized_error
                + (1.0 - self.config.ew_decay) * normalized_error
            )
            ew_absolute = (
                batch_mae
                if state.observations == 0
                else self.config.ew_decay * state.ew_absolute_error
                + (1.0 - self.config.ew_decay) * batch_mae
            )
            cusum = max(
                0.0,
                self.config.cusum_decay * state.cusum
                + surprise
                - self.config.cusum_drift,
            )
            regime_changed = cusum >= self.config.cusum_threshold
            if regime_changed:
                cusum = 0.0
            pinball_sums = list(state.pinball_sums)
            pinball_counts = list(state.pinball_counts)
            for index, level in enumerate(self.config.quantiles):
                if level in pinball:
                    pinball_sums[index] += pinball[level] * count
                    pinball_counts[index] = _checked_add(
                        pinball_counts[index], count, field="pinball_count"
                    )
            next_state = O1State(
                identity_sha256=self.identity_sha256,
                sequence=_checked_add(state.sequence, 1, field="sequence"),
                head_event_sha256=state.head_event_sha256,
                observations=_checked_add(
                    state.observations, 1, field="observations"
                ),
                value_count=_checked_add(state.value_count, count, field="value_count"),
                absolute_error_sum=state.absolute_error_sum + float(absolute.sum()),
                signed_error_sum=state.signed_error_sum + float(residual.sum()),
                pinball_sums=tuple(pinball_sums),
                pinball_counts=tuple(pinball_counts),
                interval_hits=_checked_add(
                    state.interval_hits, interval_hits, field="interval_hits"
                ),
                interval_count=_checked_add(
                    state.interval_count, interval_count, field="interval_count"
                ),
                interval_width_sum=state.interval_width_sum + interval_width_sum,
                ew_normalized_error=ew_normalized,
                ew_absolute_error=ew_absolute,
                surprise=surprise,
                learning_progress=learning_progress,
                cusum=cusum,
                regime_generation=state.regime_generation + int(regime_changed),
                regime_changes=state.regime_changes + int(regime_changed),
            )
            update = O1Update(
                observation_sha256=observation_sha256,
                sequence=next_state.sequence,
                duplicate=False,
                regime_changed=regime_changed,
                regime_generation=next_state.regime_generation,
                batch_mae=batch_mae,
                batch_bias=batch_bias,
                normalized_error=normalized_error,
                surprise=surprise,
                learning_progress=learning_progress,
                cusum=cusum,
                interval_coverage=coverage,
                pinball={level.hex(): value for level, value in pinball.items()},
            )
            committed = self._publish_event_locked(
                observation_id=selected_id,
                observation_sha256=observation_sha256,
                state=next_state,
                update=update,
            )
            self._write_state_locked(committed)
            self._state = committed
            self._observations[selected_id] = (observation_sha256, update)
            return update

    def metrics(self) -> dict[str, object]:
        with self._lock:
            state = self._state
            return {
                "bias": state.bias,
                "cusum": state.cusum,
                "ew_absolute_error": state.ew_absolute_error,
                "ew_normalized_error": state.ew_normalized_error,
                "head_event_sha256": state.head_event_sha256,
                "identity_sha256": state.identity_sha256,
                "interval_coverage": state.interval_coverage,
                "learning_progress": state.learning_progress,
                "mae": state.mae,
                "mean_interval_width": state.mean_interval_width,
                "observations": state.observations,
                "pinball": {
                    level.hex(): (
                        None
                        if state.pinball_counts[index] == 0
                        else state.pinball_sums[index]
                        / state.pinball_counts[index]
                    )
                    for index, level in enumerate(self.config.quantiles)
                },
                "regime_changes": state.regime_changes,
                "regime_generation": state.regime_generation,
                "schema": "o1fm.o1-error-metrics/v1",
                "sequence": state.sequence,
                "surprise": state.surprise,
                "value_count": state.value_count,
            }


# Concise alias used by callers that already live in the O1 namespace.
O1Stream = O1ErrorStream


__all__ = [
    "O1_CONFIG_SCHEMA",
    "O1_EVENT_SCHEMA",
    "O1_STATE_SCHEMA",
    "O1Config",
    "O1ConflictError",
    "O1Error",
    "O1ErrorStream",
    "O1IntegrityError",
    "O1State",
    "O1Stream",
    "O1Update",
]
