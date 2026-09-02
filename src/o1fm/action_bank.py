"""Economic action selection over verified forecast profiles."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType

_SCHEMA = "o1fm.action-bank/v1"
_ACTIONS = ("foundation", "crystal", "seasonal")
_MAX_TEXT_BYTES = 1024


def _canonical_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_TEXT_BYTES
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field} must be a positive bounded integer")
    return value


def _nonnegative_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{field} must be a non-negative bounded integer")
    return value


def _finite_float(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _atomic_json_dump(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    try:
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _json_load(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError("action bank state is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("action bank state must be a JSON object")
    return document


@dataclass(frozen=True, slots=True)
class ActionProfileKey:
    """One exact economic profile for a verified forecasting decision."""

    model_identity: str
    feature_schema: str
    graph_revision: str
    regime: str
    horizon: int
    target_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_identity",
            _canonical_text(self.model_identity, field="model_identity"),
        )
        object.__setattr__(
            self,
            "feature_schema",
            _canonical_text(self.feature_schema, field="feature_schema"),
        )
        object.__setattr__(
            self,
            "graph_revision",
            _canonical_text(self.graph_revision, field="graph_revision"),
        )
        object.__setattr__(
            self,
            "regime",
            _canonical_text(self.regime, field="regime"),
        )
        object.__setattr__(
            self,
            "horizon",
            _positive_int(self.horizon, field="horizon", maximum=4096),
        )
        object.__setattr__(
            self,
            "target_count",
            _positive_int(self.target_count, field="target_count", maximum=4096),
        )

    def to_document(self) -> dict[str, object]:
        return {
            "feature_schema": self.feature_schema,
            "graph_revision": self.graph_revision,
            "horizon": self.horizon,
            "model_identity": self.model_identity,
            "regime": self.regime,
            "target_count": self.target_count,
        }


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    """One settled action outcome tied to an exact profile."""

    profile: ActionProfileKey
    action: str
    quality: float
    coverage: float
    latency_seconds: float
    forward_cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ActionProfileKey):
            raise TypeError("profile must be ActionProfileKey")
        action = _canonical_text(self.action, field="action")
        if action not in _ACTIONS:
            raise ValueError("action must be foundation, crystal, or seasonal")
        coverage = _finite_float(self.coverage, field="coverage", minimum=0.0)
        if coverage > 1.0:
            raise ValueError("coverage must not exceed 1")
        object.__setattr__(self, "action", action)
        object.__setattr__(
            self, "quality", _finite_float(self.quality, field="quality")
        )
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(
            self,
            "latency_seconds",
            _finite_float(self.latency_seconds, field="latency_seconds", minimum=0.0),
        )
        object.__setattr__(
            self,
            "forward_cost",
            _finite_float(self.forward_cost, field="forward_cost", minimum=0.0),
        )

    def to_document(self) -> dict[str, object]:
        return {
            "action": self.action,
            "coverage": self.coverage,
            "forward_cost": self.forward_cost,
            "latency_seconds": self.latency_seconds,
            "profile": self.profile.to_document(),
            "quality": self.quality,
        }


@dataclass(frozen=True, slots=True)
class ActionStats:
    action: str
    support: int
    mean_quality: float
    mean_coverage: float
    mean_latency_seconds: float
    mean_forward_cost: float
    mean_total_cost: float

    def __post_init__(self) -> None:
        action = _canonical_text(self.action, field="action")
        if action not in _ACTIONS:
            raise ValueError("action must be foundation, crystal, or seasonal")
        object.__setattr__(self, "action", action)
        object.__setattr__(
            self,
            "support",
            _positive_int(self.support, field="support", maximum=10**9),
        )
        object.__setattr__(
            self,
            "mean_quality",
            _finite_float(self.mean_quality, field="mean_quality"),
        )
        coverage = _finite_float(self.mean_coverage, field="mean_coverage", minimum=0.0)
        if coverage > 1.0:
            raise ValueError("mean_coverage must not exceed 1")
        object.__setattr__(self, "mean_coverage", coverage)
        object.__setattr__(
            self,
            "mean_latency_seconds",
            _finite_float(
                self.mean_latency_seconds,
                field="mean_latency_seconds",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "mean_forward_cost",
            _finite_float(
                self.mean_forward_cost,
                field="mean_forward_cost",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "mean_total_cost",
            _finite_float(self.mean_total_cost, field="mean_total_cost", minimum=0.0),
        )


@dataclass(frozen=True, slots=True)
class ActionRecommendation:
    """One deterministic economic decision for an exact profile."""

    profile: ActionProfileKey
    action: str
    reason: str
    baseline: ActionStats | None
    chosen: ActionStats | None
    eligible_actions: Mapping[str, ActionStats]

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ActionProfileKey):
            raise TypeError("profile must be ActionProfileKey")
        action = _canonical_text(self.action, field="action")
        if action not in _ACTIONS:
            raise ValueError("action must be foundation, crystal, or seasonal")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reason", _canonical_text(self.reason, field="reason"))
        if self.baseline is not None and not isinstance(self.baseline, ActionStats):
            raise TypeError("baseline must be ActionStats or None")
        if self.chosen is not None and not isinstance(self.chosen, ActionStats):
            raise TypeError("chosen must be ActionStats or None")
        ordered: dict[str, ActionStats] = {}
        for name, stats in self.eligible_actions.items():
            canonical = _canonical_text(name, field="eligible action")
            if canonical not in _ACTIONS or not isinstance(stats, ActionStats):
                raise ValueError("eligible action entry is invalid")
            ordered[canonical] = stats
        object.__setattr__(self, "eligible_actions", MappingProxyType(ordered))


@dataclass(slots=True)
class _ActionState:
    support: int = 0
    quality_sum: float = 0.0
    coverage_sum: float = 0.0
    latency_sum: float = 0.0
    forward_cost_sum: float = 0.0
    receipts: list[dict[str, object]] | None = None

    def to_document(self) -> dict[str, object]:
        return {
            "coverage_sum": self.coverage_sum,
            "forward_cost_sum": self.forward_cost_sum,
            "latency_sum": self.latency_sum,
            "quality_sum": self.quality_sum,
            "receipts": [] if self.receipts is None else self.receipts,
            "support": self.support,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, object]) -> _ActionState:
        if not isinstance(value, Mapping):
            raise ValueError("action bank entry must be a mapping")
        support = _nonnegative_int(value.get("support"), field="support", maximum=10**9)
        quality_sum = _finite_float(value.get("quality_sum"), field="quality_sum")
        coverage_sum = _finite_float(
            value.get("coverage_sum"), field="coverage_sum", minimum=0.0
        )
        latency_sum = _finite_float(
            value.get("latency_sum"), field="latency_sum", minimum=0.0
        )
        forward_cost_sum = _finite_float(
            value.get("forward_cost_sum"),
            field="forward_cost_sum",
            minimum=0.0,
        )
        receipts_raw = value.get("receipts")
        if not isinstance(receipts_raw, list):
            raise ValueError("action bank receipts must be a list")
        receipts: list[dict[str, object]] = []
        for receipt in receipts_raw:
            if not isinstance(receipt, dict):
                raise ValueError("stored action receipt must be a mapping")
            receipts.append(dict(receipt))
        return cls(
            support=support,
            quality_sum=quality_sum,
            coverage_sum=coverage_sum,
            latency_sum=latency_sum,
            forward_cost_sum=forward_cost_sum,
            receipts=receipts,
        )


class ActionBank:
    """Select the cheapest action that preserves verified profile quality."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        quality_tolerance: float = 0.0,
        coverage_tolerance: float = 0.0,
        min_support: int = 1,
        latency_cost_weight: float = 1.0,
        forward_cost_weight: float = 1.0,
        max_receipts_per_action: int = 256,
    ) -> None:
        self._path = None if root is None else self._resolve_path(root)
        self._quality_tolerance = _finite_float(
            quality_tolerance,
            field="quality_tolerance",
            minimum=0.0,
        )
        self._coverage_tolerance = _finite_float(
            coverage_tolerance,
            field="coverage_tolerance",
            minimum=0.0,
        )
        self._min_support = _positive_int(
            min_support,
            field="min_support",
            maximum=10**6,
        )
        self._latency_cost_weight = _finite_float(
            latency_cost_weight,
            field="latency_cost_weight",
            minimum=0.0,
        )
        self._forward_cost_weight = _finite_float(
            forward_cost_weight,
            field="forward_cost_weight",
            minimum=0.0,
        )
        self._max_receipts_per_action = _positive_int(
            max_receipts_per_action,
            field="max_receipts_per_action",
            maximum=10**6,
        )
        self._profiles: dict[ActionProfileKey, dict[str, _ActionState]] = {}
        if self._path is not None and self._path.exists():
            self._load()

    @staticmethod
    def _resolve_path(root: str | os.PathLike[str]) -> Path:
        path = Path(root).expanduser().absolute()
        if path.suffix == ".json":
            return path
        return path / "action_bank.json"

    def settle(self, receipt: ActionReceipt) -> ActionStats:
        if not isinstance(receipt, ActionReceipt):
            raise TypeError("receipt must be ActionReceipt")
        actions = self._profiles.setdefault(receipt.profile, {})
        state = actions.setdefault(receipt.action, _ActionState(receipts=[]))
        state.support += 1
        state.quality_sum += receipt.quality
        state.coverage_sum += receipt.coverage
        state.latency_sum += receipt.latency_seconds
        state.forward_cost_sum += receipt.forward_cost
        assert state.receipts is not None
        state.receipts.append(receipt.to_document())
        if len(state.receipts) > self._max_receipts_per_action:
            del state.receipts[: len(state.receipts) - self._max_receipts_per_action]
        self._persist()
        return self._stats_for(receipt.action, state)

    def describe(self, profile: ActionProfileKey) -> Mapping[str, ActionStats]:
        if not isinstance(profile, ActionProfileKey):
            raise TypeError("profile must be ActionProfileKey")
        actions = self._profiles.get(profile, {})
        return MappingProxyType(
            {
                action: self._stats_for(action, state)
                for action, state in actions.items()
            }
        )

    def recommend(self, profile: ActionProfileKey) -> ActionRecommendation:
        if not isinstance(profile, ActionProfileKey):
            raise TypeError("profile must be ActionProfileKey")
        actions = self._profiles.get(profile)
        if not actions:
            return ActionRecommendation(
                profile=profile,
                action="foundation",
                reason="unknown-profile",
                baseline=None,
                chosen=None,
                eligible_actions=MappingProxyType({}),
            )
        baseline_state = actions.get("foundation")
        if baseline_state is None or baseline_state.support < self._min_support:
            return ActionRecommendation(
                profile=profile,
                action="foundation",
                reason="baseline-unverified",
                baseline=None
                if baseline_state is None
                else self._stats_for("foundation", baseline_state),
                chosen=None,
                eligible_actions=MappingProxyType({}),
            )
        baseline = self._stats_for("foundation", baseline_state)
        eligible: dict[str, ActionStats] = {"foundation": baseline}
        for action, state in actions.items():
            if action == "foundation" or state.support < self._min_support:
                continue
            stats = self._stats_for(action, state)
            if stats.mean_quality > baseline.mean_quality + self._quality_tolerance:
                continue
            if stats.mean_coverage + self._coverage_tolerance < baseline.mean_coverage:
                continue
            eligible[action] = stats
        chosen = min(
            eligible.values(),
            key=lambda stats: (
                stats.mean_total_cost,
                stats.mean_quality,
                {"foundation": 0, "seasonal": 1, "crystal": 2}[stats.action],
            ),
        )
        reason = (
            "verified-cheapest"
            if chosen.action != "foundation"
            else "foundation-cheapest"
            if len(eligible) > 1
            else "only-foundation-verified"
        )
        return ActionRecommendation(
            profile=profile,
            action=chosen.action,
            reason=reason,
            baseline=baseline,
            chosen=chosen,
            eligible_actions=MappingProxyType(eligible),
        )

    def _stats_for(self, action: str, state: _ActionState) -> ActionStats:
        if state.support <= 0:
            raise ValueError("action support must be positive")
        mean_latency = state.latency_sum / state.support
        mean_forward = state.forward_cost_sum / state.support
        total_cost = (
            self._latency_cost_weight * mean_latency
            + self._forward_cost_weight * mean_forward
        )
        return ActionStats(
            action=action,
            support=state.support,
            mean_quality=state.quality_sum / state.support,
            mean_coverage=state.coverage_sum / state.support,
            mean_latency_seconds=mean_latency,
            mean_forward_cost=mean_forward,
            mean_total_cost=total_cost,
        )

    def _persist(self) -> None:
        if self._path is None:
            return
        entries: list[dict[str, object]] = []
        for profile, actions in self._profiles.items():
            entries.append(
                {
                    "actions": {
                        action: state.to_document()
                        for action, state in sorted(actions.items())
                    },
                    "profile": profile.to_document(),
                }
            )
        _atomic_json_dump(
            self._path,
            {
                "coverage_tolerance": self._coverage_tolerance,
                "entries": entries,
                "forward_cost_weight": self._forward_cost_weight,
                "latency_cost_weight": self._latency_cost_weight,
                "max_receipts_per_action": self._max_receipts_per_action,
                "min_support": self._min_support,
                "quality_tolerance": self._quality_tolerance,
                "schema": _SCHEMA,
            },
        )

    def _load(self) -> None:
        assert self._path is not None
        document = _json_load(self._path)
        if document.get("schema") != _SCHEMA:
            raise ValueError("action bank state schema is unsupported")
        entries = document.get("entries")
        if not isinstance(entries, list):
            raise ValueError("action bank entries are invalid")
        profiles: dict[ActionProfileKey, dict[str, _ActionState]] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("action bank entry must be a mapping")
            profile = ActionProfileKey(**dict(entry["profile"]))  # type: ignore[arg-type]
            raw_actions = entry.get("actions")
            if not isinstance(raw_actions, Mapping):
                raise ValueError("action bank actions are invalid")
            actions: dict[str, _ActionState] = {}
            for action, state in raw_actions.items():
                canonical = _canonical_text(action, field="action")
                if canonical not in _ACTIONS:
                    raise ValueError("stored action name is invalid")
                actions[canonical] = _ActionState.from_document(state)
            profiles[profile] = actions
        self._profiles = profiles


__all__ = [
    "ActionBank",
    "ActionProfileKey",
    "ActionReceipt",
    "ActionRecommendation",
    "ActionStats",
]
