"""Online lagged causal routing over settled forecasting observations."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import stat
import threading
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .identity import canonical_json_bytes, require_sha256, sha256_json

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


CAUSAL_GRAPH_EVENT_SCHEMA = "o1fm.causal-graph-event/v1"
CAUSAL_GRAPH_CONFIG_SCHEMA = "o1fm.causal-graph-config/v1"
ZERO_SHA256 = "0" * 64
MAX_EVENT_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 1_000_000
MAX_VARIABLES = 4096
MAX_LAG = 4096
MAX_TOP_K = 4096

_EVENT = re.compile(r"([0-9]{20})-([0-9a-f]{64})\.json\Z")
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


class CausalGraphError(RuntimeError):
    """A graph observation, route, or persistent revision is invalid."""


class CausalGraphIntegrityError(CausalGraphError):
    """The append-only graph history failed authentication."""


class CausalGraphConflictError(CausalGraphError):
    """An observation identity was reused with different evidence."""


def _finite(value: object, *, field: str, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result) or (lower is not None and result < lower):
        raise ValueError(f"{field} is outside its finite bound")
    return 0.0 if result == 0.0 else result


def _uint(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _name(value: object, *, field: str = "variable") -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > 1024
    ):
        raise ValueError(f"{field} must be canonical bounded text")
    return value


def _float_map(value: object, *, field: str, required: bool) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    if required and not value:
        raise ValueError(f"{field} must not be empty")
    if len(value) > MAX_VARIABLES:
        raise ValueError(f"{field} exceeds its variable bound")
    result: dict[str, float] = {}
    for key, child in value.items():
        selected = _name(key, field=f"{field} key")
        result[selected] = _finite(child, field=f"{field}.{selected}")
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class CausalGraphConfig:
    decay: float = 0.98
    max_lag: int = 8
    top_k: int = 8
    min_observations: int = 8
    min_score: float = 0.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        decay = _finite(self.decay, field="decay", lower=0.0)
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        object.__setattr__(self, "decay", decay)
        object.__setattr__(
            self,
            "max_lag",
            _uint(self.max_lag, field="max_lag", minimum=1, maximum=MAX_LAG),
        )
        object.__setattr__(
            self,
            "top_k",
            _uint(self.top_k, field="top_k", minimum=1, maximum=MAX_TOP_K),
        )
        object.__setattr__(
            self,
            "min_observations",
            _uint(
                self.min_observations,
                field="min_observations",
                minimum=2,
            ),
        )
        score = _finite(self.min_score, field="min_score", lower=0.0)
        if score > 1.0:
            raise ValueError("min_score must lie in [0, 1]")
        object.__setattr__(self, "min_score", score)
        epsilon = _finite(self.epsilon, field="epsilon", lower=0.0)
        if epsilon == 0.0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "epsilon", epsilon)

    def to_record(self) -> dict[str, object]:
        return {
            "decay": self.decay.hex(),
            "epsilon": self.epsilon.hex(),
            "max_lag": self.max_lag,
            "min_observations": self.min_observations,
            "min_score": self.min_score.hex(),
            "schema": CAUSAL_GRAPH_CONFIG_SCHEMA,
            "top_k": self.top_k,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_record())


@dataclass(slots=True)
class _EWMoments:
    observations: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    covariance: float = 0.0
    variance_x: float = 0.0
    variance_y: float = 0.0

    def update(self, x: float, y: float, *, decay: float) -> None:
        if self.observations == 0:
            self.mean_x = x
            self.mean_y = y
            self.observations = 1
            return
        rate = 1.0 - decay
        dx = x - self.mean_x
        dy = y - self.mean_y
        self.covariance = decay * self.covariance + rate * dx * dy
        self.variance_x = decay * self.variance_x + rate * dx * dx
        self.variance_y = decay * self.variance_y + rate * dy * dy
        self.mean_x = decay * self.mean_x + rate * x
        self.mean_y = decay * self.mean_y + rate * y
        self.observations += 1
        if not all(
            math.isfinite(value)
            for value in (
                self.mean_x,
                self.mean_y,
                self.covariance,
                self.variance_x,
                self.variance_y,
            )
        ):
            raise CausalGraphError("EW moment update became non-finite")

    def score(self, *, epsilon: float) -> float:
        denominator = math.sqrt(
            max(0.0, self.variance_x) * max(0.0, self.variance_y) + epsilon
        )
        return min(1.0, abs(self.covariance) / denominator)


@dataclass(frozen=True, slots=True)
class CausalEdge:
    source: str
    target: str
    lag: int
    kind: Literal["observational", "intervention"]
    observations: int
    covariance: float
    source_variance: float
    target_variance: float
    score: float
    revision_sha256: str

    def __post_init__(self) -> None:
        _name(self.source, field="source")
        _name(self.target, field="target")
        _uint(self.lag, field="lag", maximum=MAX_LAG)
        if self.kind not in {"observational", "intervention"}:
            raise ValueError("edge kind is invalid")
        _uint(self.observations, field="observations", minimum=1)
        _finite(self.covariance, field="covariance")
        _finite(self.source_variance, field="source_variance", lower=0.0)
        _finite(self.target_variance, field="target_variance", lower=0.0)
        score = _finite(self.score, field="score", lower=0.0)
        if score > 1.0:
            raise ValueError("edge score exceeds one")
        require_sha256(self.revision_sha256, field="revision_sha256")

    def to_record(self) -> dict[str, object]:
        return {
            "covariance": self.covariance,
            "kind": self.kind,
            "lag": self.lag,
            "observations": self.observations,
            "revision_sha256": self.revision_sha256,
            "score": self.score,
            "source": self.source,
            "source_variance": self.source_variance,
            "target": self.target,
            "target_variance": self.target_variance,
        }


@dataclass(frozen=True, slots=True)
class GraphRevision:
    sequence: int
    event_sha256: str
    appended: bool

    def __post_init__(self) -> None:
        _uint(self.sequence, field="sequence")
        require_sha256(self.event_sha256, field="event_sha256")
        if not isinstance(self.appended, bool):
            raise TypeError("appended must be boolean")


def _thread_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.absolute()))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _graph_lock(root: Path) -> Iterator[None]:
    path = root / ".causal-graph.lock"
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_RDWR
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CausalGraphIntegrityError("graph lock must be a regular file")
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short graph event write")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class OnlineCausalGraph:
    """Append-only EW lag graph updated only from realized observations."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        graph_id: str = "default",
        config: CausalGraphConfig | None = None,
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = self.root.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CausalGraphIntegrityError("graph root must be a real directory")
        self.events = self.root / "events"
        self.events.mkdir(exist_ok=True, mode=0o700)
        if stat.S_ISLNK(self.events.lstat().st_mode):
            raise CausalGraphIntegrityError("graph events path cannot be a symlink")
        self.graph_id = _name(graph_id, field="graph_id")
        self.config = CausalGraphConfig() if config is None else config
        if not isinstance(self.config, CausalGraphConfig):
            raise TypeError("config must be CausalGraphConfig")
        self.identity_sha256 = sha256_json(
            {
                "config_sha256": self.config.sha256,
                "graph_id": self.graph_id,
                "schema": "o1fm.causal-graph-identity/v1",
            }
        )
        self._lock = _thread_lock(self.root)
        self._reset()
        with self._lock, _graph_lock(self.root):
            self._reload_locked()

    def _reset(self) -> None:
        self._sequence = 0
        self._head = ZERO_SHA256
        self._last_timestamp = -1
        self._last_values: dict[str, float] = {}
        self._last_known: dict[str, float] = {}
        self._delta_history: dict[str, list[float]] = {}
        self._moments: dict[tuple[str, str, int, str], _EWMoments] = {}
        self._observations: dict[str, str] = {}

    @property
    def revision(self) -> GraphRevision:
        with self._lock:
            return GraphRevision(self._sequence, self._head, False)

    def _event_paths(self) -> tuple[Path, ...]:
        paths = tuple(sorted(self.events.iterdir(), key=lambda item: item.name))
        if len(paths) > MAX_EVENTS:
            raise CausalGraphIntegrityError("graph event count exceeds its bound")
        if any(
            not stat.S_ISREG(path.lstat().st_mode)
            or _EVENT.fullmatch(path.name) is None
            for path in paths
        ):
            raise CausalGraphIntegrityError("graph event inventory is invalid")
        return paths

    @staticmethod
    def _read_document(path: Path) -> Mapping[str, object]:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not 0 < opened.st_size <= MAX_EVENT_BYTES
            ):
                raise CausalGraphIntegrityError("graph event size is invalid")
            raw = os.read(descriptor, opened.st_size + 1)
            if len(raw) != opened.st_size:
                raise CausalGraphIntegrityError("graph event changed while reading")
        finally:
            os.close(descriptor)
        try:
            document = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise CausalGraphIntegrityError("graph event is invalid JSON") from exc
        if not isinstance(document, Mapping) or canonical_json_bytes(document) != raw:
            raise CausalGraphIntegrityError("graph event is not canonical JSON")
        return document

    def _reload_locked(self) -> None:
        self._reset()
        previous = ZERO_SHA256
        for expected, path in enumerate(self._event_paths(), start=1):
            match = _EVENT.fullmatch(path.name)
            assert match is not None
            document = self._read_document(path)
            body = document.get("body")
            if (
                int(match.group(1)) != expected
                or set(document) != {"body", "schema", "sha256"}
                or document.get("schema") != CAUSAL_GRAPH_EVENT_SCHEMA
                or not isinstance(body, Mapping)
                or document.get("sha256") != sha256_json(body)
                or document.get("sha256") != match.group(2)
                or set(body)
                != {
                    "config_sha256",
                    "covariates",
                    "graph_identity_sha256",
                    "known_future",
                    "observation_id",
                    "observation_sha256",
                    "previous_event_sha256",
                    "sequence",
                    "targets",
                    "timestamp_ns",
                }
                or body.get("sequence") != expected
                or body.get("previous_event_sha256") != previous
                or body.get("graph_identity_sha256") != self.identity_sha256
                or body.get("config_sha256") != self.config.sha256
            ):
                raise CausalGraphIntegrityError("graph event envelope is invalid")
            observation_id = _name(body.get("observation_id"), field="observation_id")
            observation_sha256 = require_sha256(
                body.get("observation_sha256"), field="observation_sha256"
            )
            record = {
                "covariates": body["covariates"],
                "known_future": body["known_future"],
                "targets": body["targets"],
                "timestamp_ns": body["timestamp_ns"],
            }
            if observation_sha256 != sha256_json(record):
                raise CausalGraphIntegrityError("graph observation SHA-256 mismatch")
            if observation_id in self._observations:
                raise CausalGraphIntegrityError("graph repeats an observation identity")
            self._apply_record(record)
            self._sequence = expected
            self._head = document["sha256"]
            self._observations[observation_id] = observation_sha256
            previous = self._head

    def _apply_record(self, record: Mapping[str, object]) -> None:
        timestamp = _uint(record.get("timestamp_ns"), field="timestamp_ns")
        if timestamp <= self._last_timestamp:
            raise CausalGraphIntegrityError("graph timestamps must increase strictly")
        targets = _float_map(record.get("targets"), field="targets", required=True)
        covariates = _float_map(
            record.get("covariates"), field="covariates", required=False
        )
        known = _float_map(
            record.get("known_future"), field="known_future", required=False
        )
        if set(targets) & set(covariates):
            raise ValueError("target and covariate names must be disjoint")
        if set(targets) & set(known):
            raise ValueError("target and known-future names must be disjoint")
        current = {**targets, **covariates}
        deltas = {
            name: value - self._last_values[name]
            for name, value in current.items()
            if name in self._last_values
        }
        target_deltas = {name: deltas[name] for name in targets if name in deltas}
        for target, target_delta in target_deltas.items():
            for source, history in self._delta_history.items():
                for lag in range(1, min(self.config.max_lag, len(history)) + 1):
                    key = (source, target, lag, "observational")
                    self._moments.setdefault(key, _EWMoments()).update(
                        history[-lag],
                        target_delta,
                        decay=self.config.decay,
                    )
        known_deltas = {
            name: value - self._last_known[name]
            for name, value in known.items()
            if name in self._last_known
        }
        for target, target_delta in target_deltas.items():
            for source, source_delta in known_deltas.items():
                key = (source, target, 0, "intervention")
                self._moments.setdefault(key, _EWMoments()).update(
                    source_delta,
                    target_delta,
                    decay=self.config.decay,
                )
        for name, delta in deltas.items():
            history = self._delta_history.setdefault(name, [])
            history.append(delta)
            del history[: -self.config.max_lag]
        self._last_values.update(current)
        self._last_known.update(known)
        self._last_timestamp = timestamp

    def _publish_locked(
        self,
        *,
        observation_id: str,
        observation_sha256: str,
        record: Mapping[str, object],
        sync_directory: bool = True,
    ) -> str:
        sequence = self._sequence + 1
        body = {
            "config_sha256": self.config.sha256,
            "covariates": record["covariates"],
            "graph_identity_sha256": self.identity_sha256,
            "known_future": record["known_future"],
            "observation_id": observation_id,
            "observation_sha256": observation_sha256,
            "previous_event_sha256": self._head,
            "sequence": sequence,
            "targets": record["targets"],
            "timestamp_ns": record["timestamp_ns"],
        }
        digest = sha256_json(body)
        data = canonical_json_bytes(
            {"body": body, "schema": CAUSAL_GRAPH_EVENT_SCHEMA, "sha256": digest}
        )
        if len(data) > MAX_EVENT_BYTES:
            raise CausalGraphError("graph event exceeds its byte bound")
        destination = self.events / f"{sequence:020d}-{digest}.json"
        temporary = self.events / f".{destination.name}.{secrets.token_hex(8)}.tmp"
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
            if sync_directory:
                _fsync_directory(self.events)
        except FileExistsError as exc:
            raise CausalGraphConflictError("graph revision already exists") from exc
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    @staticmethod
    def _prepare_observation(
        *,
        timestamp_ns: int,
        targets: Mapping[str, float],
        covariates: Mapping[str, float] | None,
        known_future: Mapping[str, float] | None,
        observation_id: str | None,
    ) -> tuple[dict[str, object], str, str]:
        record: dict[str, object] = {
            "covariates": _float_map(
                {} if covariates is None else covariates,
                field="covariates",
                required=False,
            ),
            "known_future": _float_map(
                {} if known_future is None else known_future,
                field="known_future",
                required=False,
            ),
            "targets": _float_map(targets, field="targets", required=True),
            "timestamp_ns": _uint(timestamp_ns, field="timestamp_ns"),
        }
        if set(record["targets"]) & set(record["covariates"]):  # type: ignore[arg-type]
            raise ValueError("target and covariate names must be disjoint")
        if set(record["targets"]) & set(record["known_future"]):  # type: ignore[arg-type]
            raise ValueError("target and known-future names must be disjoint")
        observation_sha256 = sha256_json(record)
        selected_id = (
            f"sha256:{observation_sha256}"
            if observation_id is None
            else _name(observation_id, field="observation_id")
        )
        return record, selected_id, observation_sha256

    def observe_many(
        self,
        observations: Sequence[Mapping[str, object]],
    ) -> tuple[GraphRevision, ...]:
        """Append a validated temporal batch with one history reload and fsync."""

        if isinstance(observations, (str, bytes)) or not isinstance(
            observations, Sequence
        ):
            raise TypeError("observations must be a sequence of mappings")
        if len(observations) > MAX_EVENTS:
            raise ValueError("observation batch exceeds the event bound")
        prepared: list[tuple[dict[str, object], str, str]] = []
        fields = {
            "covariates",
            "known_future",
            "observation_id",
            "targets",
            "timestamp_ns",
        }
        for raw in observations:
            if (
                not isinstance(raw, Mapping)
                or not {
                    "targets",
                    "timestamp_ns",
                }.issubset(raw)
                or not set(raw).issubset(fields)
            ):
                raise ValueError("graph batch observation fields are invalid")
            prepared.append(
                self._prepare_observation(
                    timestamp_ns=raw["timestamp_ns"],  # type: ignore[arg-type]
                    targets=raw["targets"],  # type: ignore[arg-type]
                    covariates=raw.get("covariates"),  # type: ignore[arg-type]
                    known_future=raw.get("known_future"),  # type: ignore[arg-type]
                    observation_id=raw.get("observation_id"),  # type: ignore[arg-type]
                )
            )
        if not prepared:
            return ()

        with self._lock, _graph_lock(self.root):
            self._reload_locked()
            last_timestamp = self._last_timestamp
            pending: dict[str, str] = {}
            append_flags: list[bool] = []
            for record, selected_id, observation_sha256 in prepared:
                prior = self._observations.get(selected_id, pending.get(selected_id))
                if prior is not None:
                    if prior != observation_sha256:
                        raise CausalGraphConflictError(
                            "observation_id is bound to different graph evidence"
                        )
                    append_flags.append(False)
                    continue
                timestamp = int(record["timestamp_ns"])
                if timestamp <= last_timestamp:
                    raise ValueError("timestamp_ns must increase strictly")
                last_timestamp = timestamp
                pending[selected_id] = observation_sha256
                append_flags.append(True)

            revisions: list[GraphRevision] = []
            appended_any = False
            try:
                for should_append, prepared_row in zip(
                    append_flags,
                    prepared,
                    strict=True,
                ):
                    record, selected_id, observation_sha256 = prepared_row
                    if not should_append:
                        revisions.append(
                            GraphRevision(self._sequence, self._head, False)
                        )
                        continue
                    digest = self._publish_locked(
                        observation_id=selected_id,
                        observation_sha256=observation_sha256,
                        record=record,
                        sync_directory=False,
                    )
                    self._apply_record(record)
                    self._sequence += 1
                    self._head = digest
                    self._observations[selected_id] = observation_sha256
                    appended_any = True
                    revisions.append(GraphRevision(self._sequence, self._head, True))
            finally:
                if appended_any:
                    _fsync_directory(self.events)
            return tuple(revisions)

    def observe(
        self,
        *,
        timestamp_ns: int,
        targets: Mapping[str, float],
        covariates: Mapping[str, float] | None = None,
        known_future: Mapping[str, float] | None = None,
        observation_id: str | None = None,
    ) -> GraphRevision:
        """Append one realized time step and update lag/intervention moments."""

        return self.observe_many(
            (
                {
                    "covariates": {} if covariates is None else covariates,
                    "known_future": {} if known_future is None else known_future,
                    "observation_id": observation_id,
                    "targets": targets,
                    "timestamp_ns": timestamp_ns,
                },
            )
        )[0]

    def all_edges(self, *, target: str | None = None) -> tuple[CausalEdge, ...]:
        """Return every measured edge, including still-under-supported edges."""

        selected_target = None if target is None else _name(target, field="target")
        with self._lock:
            rows = []
            for (source, edge_target, lag, kind), moments in self._moments.items():
                if selected_target is not None and edge_target != selected_target:
                    continue
                rows.append(
                    CausalEdge(
                        source=source,
                        target=edge_target,
                        lag=lag,
                        kind=kind,  # type: ignore[arg-type]
                        observations=moments.observations,
                        covariance=moments.covariance,
                        source_variance=max(0.0, moments.variance_x),
                        target_variance=max(0.0, moments.variance_y),
                        score=moments.score(epsilon=self.config.epsilon),
                        revision_sha256=self._head,
                    )
                )
            return tuple(
                sorted(
                    rows,
                    key=lambda edge: (
                        edge.target,
                        -edge.score,
                        edge.kind,
                        edge.source,
                        edge.lag,
                    ),
                )
            )

    def route(self, target: str, *, top_k: int | None = None) -> tuple[CausalEdge, ...]:
        """Return deterministic strongest supported incoming edges for a target."""

        limit = (
            self.config.top_k
            if top_k is None
            else _uint(top_k, field="top_k", minimum=1, maximum=MAX_TOP_K)
        )
        eligible = tuple(
            edge
            for edge in self.all_edges(target=target)
            if edge.observations >= self.config.min_observations
            and edge.score >= self.config.min_score
        )
        return eligible[:limit]

    routed_edges = route

    def metrics(self) -> dict[str, object]:
        with self._lock:
            supported = sum(
                moments.observations >= self.config.min_observations
                and moments.score(epsilon=self.config.epsilon) >= self.config.min_score
                for moments in self._moments.values()
            )
            return {
                "edge_count": len(self._moments),
                "graph_identity_sha256": self.identity_sha256,
                "head_event_sha256": self._head,
                "last_timestamp_ns": (
                    None if self._last_timestamp < 0 else self._last_timestamp
                ),
                "revision": self._sequence,
                "schema": "o1fm.causal-graph-metrics/v1",
                "supported_edges": supported,
            }


CausalGraph = OnlineCausalGraph


__all__ = [
    "CAUSAL_GRAPH_CONFIG_SCHEMA",
    "CAUSAL_GRAPH_EVENT_SCHEMA",
    "CausalEdge",
    "CausalGraph",
    "CausalGraphConfig",
    "CausalGraphConflictError",
    "CausalGraphError",
    "CausalGraphIntegrityError",
    "GraphRevision",
    "OnlineCausalGraph",
]
