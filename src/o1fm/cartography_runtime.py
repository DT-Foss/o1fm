"""Execution bridge from forecast cartography cells to branch forecasts.

The cartographer reasons in small horizon/target cells, while a forecasting
model naturally emits a whole trajectory.  This module coalesces compatible
cells into one physical branch forward, derives independently scored evidence
for every cell, and charges the physical work exactly once across the batch.

Probe scoring deliberately owns a private in-memory error stream.  It never
calls :class:`~o1fm.o1.O1ErrorStream` or
:meth:`~o1fm.runtime.O1FMRuntime.settle`, so exploratory cartography cannot
advance the live authority state.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, TypeVar

import numpy as np

from .branches import (
    BranchExecutionResult,
    InterventionPlan,
    JointEnvelope,
    apply_intervention,
)
from .cartography import ZERO_SHA256, ForecastProbeTarget, ProbeEvidence
from .contracts import (
    ForecastExample,
    ForecastInputs,
    ForecastPrediction,
    ForecastTarget,
)
from .identity import canonical_json_bytes, require_sha256, sha256_bytes, sha256_json
from .metrics import score_prediction

_EXECUTION_KEY_SCHEMA = "o1fm.cartography-execution-key/v1"
_EXECUTION_PAYLOAD_SCHEMA = "o1fm.cartography-execution-payload/v1"
_CELL_RESULT_SCHEMA = "o1fm.cartography-cell-result/v1"
_ATLAS_CITATIONS_SCHEMA = "o1fm.cartography-atlas-citations/v1"
_MAX_REGISTERED_TARGETS = 1_000_000


class CartographyExecutionError(RuntimeError):
    """A physical forecast cannot satisfy its declared probe cells."""


class CartographyExecutionContractError(CartographyExecutionError, ValueError):
    """A target, window, intervention, or branch result is incompatible."""


class CartographyExecutionBudgetError(CartographyExecutionError):
    """Measured work exceeds the coalesced targets' declared work capacity."""


class CartographyProbeNotReadyError(CartographyExecutionError):
    """A probe cell has no realized value and must remain pending."""


def _finite(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{field} must be finite and non-negative")
    return 0.0 if result == 0.0 else result


def _positive_integer(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def execution_key(target: ForecastProbeTarget) -> str:
    """Return the physical-forward identity shared by compatible probe cells.

    Horizon, target subset, and accounting reservations are intentionally not
    part of this key: one full trajectory can satisfy all of those logical
    cells.  Every input capable of changing the physical execution remains in
    the identity.
    """

    if not isinstance(target, ForecastProbeTarget):
        raise TypeError("target must be ForecastProbeTarget")
    return sha256_json(
        {
            "action_inventory": list(target.action_inventory),
            "dataset_identity_sha256": target.dataset_identity_sha256,
            "graph_topology_sha256": target.graph_topology_sha256,
            "intervention_identity_sha256": target.intervention_identity_sha256,
            "model_identity_sha256": target.model_identity_sha256,
            "regime": target.regime,
            "schema": _EXECUTION_KEY_SCHEMA,
            "seed": target.seed,
            "series_id": target.series_id,
        }
    )


@dataclass(frozen=True, slots=True)
class ProbeWindow:
    """Leakage-safe inputs and explicitly scenario-bound realized labels.

    Factual labels use the zero identity.  Simulated, experimental, or oracle
    labels for an intervention must carry that exact plan identity, preventing
    an intervention forecast from being silently scored against factual truth.
    """

    inputs: ForecastInputs
    target: ForecastTarget
    interventions: Sequence[InterventionPlan] = ()
    realized_intervention_identity_sha256: str = ZERO_SHA256

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, ForecastInputs):
            raise TypeError("inputs must be ForecastInputs")
        if not isinstance(self.target, ForecastTarget):
            raise TypeError("target must be ForecastTarget")
        ForecastExample(self.inputs, self.target)
        realized_identity = require_sha256(
            self.realized_intervention_identity_sha256,
            field="realized_intervention_identity_sha256",
        )
        if isinstance(self.interventions, (str, bytes)) or not isinstance(
            self.interventions, Sequence
        ):
            raise TypeError("interventions must be a sequence")
        plans = tuple(self.interventions)
        if not all(isinstance(plan, InterventionPlan) for plan in plans):
            raise TypeError("interventions must contain InterventionPlan instances")
        by_identity = {plan.identity: plan for plan in plans}
        if len(by_identity) != len(plans):
            raise ValueError("intervention identities must be unique")
        if realized_identity != ZERO_SHA256 and realized_identity not in by_identity:
            raise ValueError(
                "realized intervention identity requires its InterventionPlan"
            )
        object.__setattr__(
            self,
            "interventions",
            tuple(by_identity[key] for key in sorted(by_identity)),
        )
        object.__setattr__(
            self,
            "realized_intervention_identity_sha256",
            realized_identity,
        )

    def intervention(self, identity_sha256: str) -> InterventionPlan | None:
        if identity_sha256 == ZERO_SHA256:
            return None
        for plan in self.interventions:
            if plan.identity == identity_sha256:
                return plan
        raise CartographyExecutionContractError(
            "probe window does not contain the target intervention"
        )


@dataclass(frozen=True, slots=True)
class MeasuredBranchExecution:
    """One branch result with its measured physical execution cost."""

    result: BranchExecutionResult
    work_cost: float
    forward_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.result, BranchExecutionResult):
            raise TypeError("result must be BranchExecutionResult")
        object.__setattr__(
            self,
            "work_cost",
            _finite(self.work_cost, field="work_cost", nonnegative=True),
        )
        object.__setattr__(
            self,
            "forward_count",
            _positive_integer(self.forward_count, field="forward_count"),
        )


class BranchProbeCallable(Protocol):
    def __call__(
        self, inputs: ForecastInputs
    ) -> BranchExecutionResult | MeasuredBranchExecution: ...


WindowResolver = Callable[[ForecastProbeTarget], ProbeWindow]
_TransactionResult = TypeVar("_TransactionResult")


@dataclass(frozen=True, slots=True)
class ShadowProbeUpdate:
    """One non-authoritative, process-local O1 comparison."""

    normalized_error: float
    surprise: float
    learning_progress: float
    previous_error: float


class ShadowProbeLedger:
    """Small private error stream used only to value cartography evidence."""

    def __init__(self, *, ew_decay: float = 0.95) -> None:
        decay = _finite(ew_decay, field="ew_decay", nonnegative=True)
        if decay >= 1.0:
            raise ValueError("ew_decay must be inside [0, 1)")
        self.ew_decay = decay
        self._errors: dict[str, float] = {}
        self._observations: dict[str, int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _identity(identity_sha256: str) -> str:
        if (
            not isinstance(identity_sha256, str)
            or len(identity_sha256) != 64
            or any(character not in "0123456789abcdef" for character in identity_sha256)
        ):
            raise ValueError("shadow identity must be a SHA-256 digest")
        return identity_sha256

    def _transaction(
        self,
        entries: Sequence[tuple[str, float]],
        build: Callable[[tuple[ShadowProbeUpdate, ...]], _TransactionResult],
    ) -> _TransactionResult:
        """Build a result and commit all shadow updates as one transaction."""

        prepared = tuple(
            (
                self._identity(identity),
                _finite(error, field="normalized_error", nonnegative=True),
            )
            for identity, error in entries
        )
        with self._lock:
            errors = dict(self._errors)
            observations = dict(self._observations)
            updates: list[ShadowProbeUpdate] = []
            for identity, current in prepared:
                previous = errors.get(identity, current)
                surprise = max(0.0, current - previous)
                learning = max(0.0, previous - current)
                updated = (
                    current
                    if identity not in errors
                    else self.ew_decay * previous + (1.0 - self.ew_decay) * current
                )
                errors[identity] = updated
                observations[identity] = observations.get(identity, 0) + 1
                updates.append(
                    ShadowProbeUpdate(
                        normalized_error=current,
                        surprise=surprise,
                        learning_progress=learning,
                        previous_error=previous,
                    )
                )
            result = build(tuple(updates))
            self._errors = errors
            self._observations = observations
            return result

    def observe(
        self, identity_sha256: str, normalized_error: float
    ) -> ShadowProbeUpdate:
        return self._transaction(
            ((identity_sha256, normalized_error),),
            lambda updates: updates[0],
        )

    def snapshot(self) -> Mapping[str, Mapping[str, float | int]]:
        with self._lock:
            return MappingProxyType(
                {
                    identity: MappingProxyType(
                        {
                            "ew_normalized_error": self._errors[identity],
                            "observations": self._observations[identity],
                        }
                    )
                    for identity in sorted(self._errors)
                }
            )


@dataclass(frozen=True, slots=True)
class CoalescedProbeBatch:
    """Evidence and exact physical accounting for one execution key."""

    execution_key_sha256: str
    execution_result_sha256: str
    target_sha256s: tuple[str, ...]
    evidence: Mapping[str, ProbeEvidence]
    work_charges: Mapping[str, float]
    byte_charges: Mapping[str, int]
    physical_work: float
    payload_bytes: int
    ledger_bytes: int
    forward_count: int
    reservation_shortfall_bytes: int = 0

    def __post_init__(self) -> None:
        targets = tuple(sorted(self.target_sha256s))
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("target_sha256s must be non-empty and unique")
        if set(self.evidence) != set(targets):
            raise ValueError("evidence must cover every target exactly once")
        if set(self.work_charges) != set(targets):
            raise ValueError("work charges must cover every target exactly once")
        if set(self.byte_charges) != set(targets):
            raise ValueError("byte charges must cover every target exactly once")
        physical = _finite(
            self.physical_work,
            field="physical_work",
            nonnegative=True,
        )
        work = {
            key: _finite(value, field="work charge", nonnegative=True)
            for key, value in self.work_charges.items()
        }
        byte_charges = {key: int(value) for key, value in self.byte_charges.items()}
        if any(value < 0 for value in byte_charges.values()):
            raise ValueError("byte charges must be non-negative")
        if not math.isclose(sum(work.values()), physical, abs_tol=1e-12):
            raise ValueError("work charges do not equal physical work")
        if sum(byte_charges.values()) != self.payload_bytes:
            raise ValueError("byte charges do not equal payload bytes")
        if (
            isinstance(self.ledger_bytes, bool)
            or not isinstance(self.ledger_bytes, int)
            or self.ledger_bytes < 0
        ):
            raise ValueError("ledger_bytes must be a non-negative integer")
        object.__setattr__(self, "target_sha256s", targets)
        object.__setattr__(
            self,
            "evidence",
            MappingProxyType(dict(sorted(self.evidence.items()))),
        )
        object.__setattr__(
            self,
            "work_charges",
            MappingProxyType(dict(sorted(work.items()))),
        )
        object.__setattr__(
            self,
            "byte_charges",
            MappingProxyType(dict(sorted(byte_charges.items()))),
        )

    def evidence_for(self, target: ForecastProbeTarget | str) -> ProbeEvidence:
        identity = target.sha256 if isinstance(target, ForecastProbeTarget) else target
        try:
            return self.evidence[identity]
        except KeyError as exc:
            raise KeyError("target is not part of this coalesced batch") from exc


@dataclass(frozen=True, slots=True)
class _CachedForward:
    key_sha256: str
    window: ProbeWindow
    result: BranchExecutionResult
    result_sha256: str
    payload_bytes: int
    work_cost: float
    forward_count: int
    started_at: float
    completed_at: float


@dataclass(frozen=True, slots=True)
class _PendingEvidence:
    target: ForecastProbeTarget
    coverage_gap: float
    pinball: float
    mae: float
    joint_miss: float
    normalized_error: float
    route_w1: float
    observed_values: int
    result_sha256: str
    atlas_sha256: str
    started_at: float
    completed_at: float
    shadow_identity_sha256: str


def _array_blob(value: object) -> tuple[dict[str, object], bytes]:
    raw = np.asarray(value)
    dtype = raw.dtype.newbyteorder("<")
    array = np.ascontiguousarray(raw, dtype=dtype)
    blob = array.tobytes(order="C")
    return (
        {
            "data_sha256": sha256_bytes(blob),
            "dtype": array.dtype.str,
            "shape": [int(axis) for axis in array.shape],
        },
        blob,
    )


def _citation_record(result: BranchExecutionResult) -> list[dict[str, object]]:
    return [
        {
            "kind": receipt.kind,
            "record_count": receipt.record_count,
            "revision_sequence": receipt.revision_sequence,
            "revision_sha256": receipt.revision_sha256,
            "segment_sha256": receipt.segment_sha256,
        }
        for receipt in result.atlas_citations
    ]


def _execution_payload(result: BranchExecutionResult) -> tuple[str, int]:
    arrays: list[bytes] = []

    def describe(value: object) -> dict[str, object]:
        descriptor, blob = _array_blob(value)
        arrays.append(blob)
        return descriptor

    prediction = result.prediction
    metadata: dict[str, object] = {
        "action_identity": result.action_identity,
        "assumptions": list(result.assumptions),
        "atlas_citations": _citation_record(result),
        "graph_identity": result.graph_identity,
        "model_identity": result.model_identity,
        "prediction": {
            "point": describe(prediction.point),
            "quantile_levels": (
                None
                if prediction.quantile_levels is None
                else describe(prediction.quantile_levels)
            ),
            "quantiles": (
                None if prediction.quantiles is None else describe(prediction.quantiles)
            ),
            "series_id": prediction.series_id,
            "timestamps": describe(prediction.timestamps.astype(np.int64, copy=False)),
        },
        "profile_identity": result.profile_identity,
        "route_edges": [edge.to_record() for edge in result.route_edges],
        "route_identity": result.route_identity,
        "route_w1": result.route_w1.hex(),
        "schema": _EXECUTION_PAYLOAD_SCHEMA,
    }
    if result.joint_envelope is None:
        metadata["joint_envelope"] = None
    else:
        metadata["joint_envelope"] = {
            "lower": describe(result.joint_envelope.lower),
            "timestamps": describe(
                result.joint_envelope.timestamps.astype(np.int64, copy=False)
            ),
            "upper": describe(result.joint_envelope.upper),
        }
    manifest = canonical_json_bytes(metadata)
    digest = hashlib.sha256()
    payload_bytes = 0
    for blob in (manifest, *arrays):
        frame = len(blob).to_bytes(8, byteorder="big", signed=False)
        digest.update(frame)
        digest.update(blob)
        payload_bytes += len(frame) + len(blob)
    return digest.hexdigest(), payload_bytes


def _prefix_inputs(inputs: ForecastInputs, horizon: int) -> ForecastInputs:
    if horizon == inputs.horizon:
        return inputs
    rows = inputs.history_length + horizon
    return ForecastInputs(
        series_id=inputs.series_id,
        horizon=horizon,
        timestamps=inputs.timestamps[:rows],
        past_target=inputs.past_target,
        past_target_mask=inputs.past_target_mask,
        past_covariates=inputs.past_covariates,
        past_covariate_mask=inputs.past_covariate_mask,
        known_covariates=inputs.known_covariates[:rows],
        known_covariate_mask=inputs.known_covariate_mask[:rows],
    )


def _slice_prediction(
    prediction: ForecastPrediction,
    target: ForecastProbeTarget,
) -> ForecastPrediction:
    columns = list(target.target_indices)
    quantiles = (
        None
        if prediction.quantiles is None
        else prediction.quantiles[: target.horizon, columns, :]
    )
    return ForecastPrediction(
        series_id=prediction.series_id,
        timestamps=prediction.timestamps[: target.horizon],
        point=prediction.point[: target.horizon, columns],
        quantiles=quantiles,
        quantile_levels=prediction.quantile_levels,
    )


def _slice_target(window: ProbeWindow, target: ForecastProbeTarget) -> ForecastTarget:
    columns = list(target.target_indices)
    return ForecastTarget(
        series_id=window.target.series_id,
        timestamps=window.target.timestamps[: target.horizon],
        values=window.target.values[: target.horizon, columns],
        mask=window.target.mask[: target.horizon, columns],
    )


def _slice_envelope(
    envelope: JointEnvelope | None,
    target: ForecastProbeTarget,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if envelope is None:
        return None, None
    columns = list(target.target_indices)
    return (
        envelope.lower[: target.horizon, columns],
        envelope.upper[: target.horizon, columns],
    )


def _same_model_prefix(left: ProbeWindow, right: ProbeWindow, horizon: int) -> bool:
    left_rows = left.inputs.history_length + horizon
    right_rows = right.inputs.history_length + horizon
    return bool(
        left.inputs.series_id == right.inputs.series_id
        and left.inputs.history_length == right.inputs.history_length
        and left.inputs.target_dim == right.inputs.target_dim
        and np.array_equal(left.inputs.past_target, right.inputs.past_target)
        and np.array_equal(left.inputs.past_target_mask, right.inputs.past_target_mask)
        and np.array_equal(left.inputs.past_covariates, right.inputs.past_covariates)
        and np.array_equal(
            left.inputs.past_covariate_mask,
            right.inputs.past_covariate_mask,
        )
        and np.array_equal(
            left.inputs.timestamps[:left_rows],
            right.inputs.timestamps[:right_rows],
        )
        and np.array_equal(
            left.inputs.known_covariates[:left_rows],
            right.inputs.known_covariates[:right_rows],
        )
        and np.array_equal(
            left.inputs.known_covariate_mask[:left_rows],
            right.inputs.known_covariate_mask[:right_rows],
        )
    )


def _allocate_work(
    total: float,
    targets: Sequence[ForecastProbeTarget],
) -> dict[str, float]:
    remaining = total
    result = {target.sha256: 0.0 for target in targets}
    for target in sorted(targets, key=lambda item: item.sha256):
        charge = min(target.expected_work, remaining)
        result[target.sha256] = charge
        remaining -= charge
        if remaining <= 1e-12:
            remaining = 0.0
            break
    if remaining > 0.0:
        raise CartographyExecutionBudgetError(
            "measured physical work exceeds coalesced expected_work"
        )
    # Make the equality exact in binary floating point for downstream sums.
    owner = min(result)
    result[owner] += total - sum(result.values())
    return result


def _allocate_bytes(
    total: int,
    targets: Sequence[ForecastProbeTarget],
) -> tuple[dict[str, int], int]:
    remaining = total
    result = {target.sha256: 0 for target in targets}
    ordered = sorted(targets, key=lambda item: item.sha256)
    for target in ordered:
        charge = min(target.expected_bytes, remaining)
        result[target.sha256] = charge
        remaining -= charge
        if remaining == 0:
            break
    shortfall = remaining
    if remaining:
        result[ordered[0].sha256] += remaining
    return result, shortfall


def _shadow_identity(
    target: ForecastProbeTarget,
    result: BranchExecutionResult,
) -> str:
    return sha256_json(
        {
            "action_identity": result.action_identity,
            "action_inventory": list(target.action_inventory),
            "graph_topology_sha256": target.graph_topology_sha256,
            "horizon": target.horizon,
            "intervention_identity_sha256": target.intervention_identity_sha256,
            "model_identity_sha256": target.model_identity_sha256,
            "regime": target.regime,
            "schema": "o1fm.cartography-shadow-cell/v1",
            "series_id": target.series_id,
            "target_indices": list(target.target_indices),
        }
    )


def _normalized_error(
    prediction: ForecastPrediction,
    actual: ForecastTarget,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> float:
    observed = np.asarray(actual.mask, dtype=np.bool_)
    residual = np.abs(
        np.asarray(actual.values, dtype=np.float64)
        - np.asarray(prediction.point, dtype=np.float64)
    )
    if lower is None or upper is None:
        scale = np.ones(actual.values.shape, dtype=np.float64)
    else:
        scale = np.maximum(
            (np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64))
            * 0.5,
            1e-8,
        )
    return float(np.mean(residual[observed] / (scale[observed] + 1e-8)))


class CartographyExecutionBridge:
    """Thread-safe synchronous executor accepted directly by a cartographer.

    Registering a frontier before passing this object to ``run`` or
    ``background_run`` enables deterministic cross-cell coalescing.  Direct
    calls and :meth:`execute_many` are synchronous and use the same cache.
    """

    def __init__(
        self,
        resolve_window: WindowResolver,
        forecast: BranchProbeCallable,
        *,
        nominal_coverage: float = 0.8,
        default_work_cost: float = 1.0,
        shadow: ShadowProbeLedger | None = None,
        clock: Callable[[], float] = time.time,
        strict_byte_reservations: bool = False,
    ) -> None:
        if not callable(resolve_window):
            raise TypeError("resolve_window must be callable")
        if not callable(forecast):
            raise TypeError("forecast must be callable")
        coverage = _finite(
            nominal_coverage,
            field="nominal_coverage",
            nonnegative=True,
        )
        if coverage > 1.0:
            raise ValueError("nominal_coverage must be inside [0, 1]")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not isinstance(strict_byte_reservations, bool):
            raise TypeError("strict_byte_reservations must be boolean")
        self.resolve_window = resolve_window
        self.forecast = forecast
        self.nominal_coverage = coverage
        self.default_work_cost = _finite(
            default_work_cost,
            field="default_work_cost",
            nonnegative=True,
        )
        self.shadow = ShadowProbeLedger() if shadow is None else shadow
        if not isinstance(self.shadow, ShadowProbeLedger):
            raise TypeError("shadow must be ShadowProbeLedger")
        self.clock = clock
        self.strict_byte_reservations = strict_byte_reservations
        self._registered: dict[str, ForecastProbeTarget] = {}
        self._evidence: dict[str, ProbeEvidence] = {}
        self._forwards: dict[str, _CachedForward] = {}
        self._batches: list[CoalescedProbeBatch] = []
        self._lock = threading.RLock()

    def register(self, targets: Sequence[ForecastProbeTarget]) -> int:
        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
            raise TypeError("targets must be a sequence")
        selected = tuple(targets)
        if not all(isinstance(target, ForecastProbeTarget) for target in selected):
            raise TypeError("targets must contain ForecastProbeTarget instances")
        with self._lock:
            added = 0
            for target in selected:
                previous = self._registered.get(target.sha256)
                if previous is None:
                    if len(self._registered) >= _MAX_REGISTERED_TARGETS:
                        raise CartographyExecutionContractError(
                            "registered target bound was reached"
                        )
                    self._registered[target.sha256] = target
                    added += 1
                elif previous != target:  # Defensive against an identity collision.
                    raise CartographyExecutionContractError(
                        "target identity collision has incompatible contracts"
                    )
            return added

    @property
    def batches(self) -> tuple[CoalescedProbeBatch, ...]:
        with self._lock:
            return tuple(self._batches)

    def metrics(self) -> Mapping[str, float | int]:
        with self._lock:
            return MappingProxyType(
                {
                    "batches": len(self._batches),
                    "cached_evidence": len(self._evidence),
                    "execution_keys": len(self._forwards),
                    "forward_count": sum(
                        batch.forward_count for batch in self._batches
                    ),
                    "payload_bytes": sum(
                        batch.payload_bytes for batch in self._batches
                    ),
                    "ledger_bytes": sum(batch.ledger_bytes for batch in self._batches),
                    "physical_work": sum(
                        batch.physical_work for batch in self._batches
                    ),
                    "reservation_shortfall_bytes": sum(
                        batch.reservation_shortfall_bytes for batch in self._batches
                    ),
                }
            )

    def _resolve(self, target: ForecastProbeTarget) -> ProbeWindow:
        window = self.resolve_window(target)
        if not isinstance(window, ProbeWindow):
            raise TypeError("resolve_window must return ProbeWindow")
        if target.series_id != window.inputs.series_id:
            raise CartographyExecutionContractError(
                "target series_id differs from its resolved window"
            )
        if (
            target.intervention_identity_sha256
            != window.realized_intervention_identity_sha256
        ):
            raise CartographyExecutionContractError(
                "probe intervention differs from its realized/oracle target labels"
            )
        if target.horizon > window.inputs.horizon:
            raise CartographyExecutionContractError(
                "probe horizon exceeds its resolved window"
            )
        if max(target.target_indices) >= window.inputs.target_dim:
            raise CartographyExecutionContractError(
                "probe target index exceeds its resolved window"
            )
        observed = window.target.mask[
            : target.horizon,
            list(target.target_indices),
        ]
        if not bool(np.asarray(observed, dtype=np.bool_).any()):
            raise CartographyProbeNotReadyError(
                "probe cell has no realized target value"
            )
        window.intervention(target.intervention_identity_sha256)
        return window

    def _physical_forward(
        self,
        key_sha256: str,
        targets: Sequence[ForecastProbeTarget],
        windows: Mapping[str, ProbeWindow],
    ) -> _CachedForward:
        horizon = max(target.horizon for target in targets)
        candidates = sorted(
            (windows[target.sha256] for target in targets),
            key=lambda window: (-window.inputs.horizon, window.inputs.series_id),
        )
        selected = candidates[0]
        for target in targets:
            window = windows[target.sha256]
            if not _same_model_prefix(selected, window, target.horizon):
                raise CartographyExecutionContractError(
                    "coalesced probe windows do not share one exact model prefix"
                )
        intervention_identity = targets[0].intervention_identity_sha256
        plan = selected.intervention(intervention_identity)
        execution_inputs = selected.inputs
        if plan is not None:
            execution_inputs = apply_intervention(execution_inputs, plan)
        execution_inputs = _prefix_inputs(execution_inputs, horizon)
        started = _finite(self.clock(), field="started_at", nonnegative=True)
        raw = self.forecast(execution_inputs)
        completed = _finite(self.clock(), field="completed_at", nonnegative=True)
        if completed < started:
            raise CartographyExecutionContractError("execution clock moved backwards")
        measured = (
            raw
            if isinstance(raw, MeasuredBranchExecution)
            else MeasuredBranchExecution(raw, self.default_work_cost)
        )
        measured.result.prediction.validate_for(execution_inputs)
        if measured.result.joint_envelope is not None:
            measured.result.joint_envelope.validate_for(measured.result.prediction)
        result_sha256, payload_bytes = _execution_payload(measured.result)
        return _CachedForward(
            key_sha256=key_sha256,
            window=selected,
            result=measured.result,
            result_sha256=result_sha256,
            payload_bytes=payload_bytes,
            work_cost=measured.work_cost,
            forward_count=measured.forward_count,
            started_at=started,
            completed_at=completed,
        )

    def _score_cell(
        self,
        target: ForecastProbeTarget,
        window: ProbeWindow,
        cached: _CachedForward,
    ) -> _PendingEvidence:
        prediction = _slice_prediction(cached.result.prediction, target)
        actual = _slice_target(window, target)
        score = score_prediction(prediction, actual)
        envelope_lower, envelope_upper = _slice_envelope(
            cached.result.joint_envelope,
            target,
        )
        if envelope_lower is None or envelope_upper is None:
            if prediction.quantiles is None:
                lower = prediction.point
                upper = prediction.point
            else:
                lower = prediction.quantiles[..., 0]
                upper = prediction.quantiles[..., -1]
        else:
            lower = envelope_lower
            upper = envelope_upper
        observed = np.asarray(actual.mask, dtype=np.bool_)
        hits = (np.asarray(actual.values)[observed] >= np.asarray(lower)[observed]) & (
            np.asarray(actual.values)[observed] <= np.asarray(upper)[observed]
        )
        joint_miss = float(not bool(hits.all()))
        normalized_error = _normalized_error(prediction, actual, lower, upper)
        coverage = 0.0 if score.interval_coverage is None else score.interval_coverage
        result_sha256 = sha256_json(
            {
                "execution_result_sha256": cached.result_sha256,
                "prediction": {
                    "point": _array_blob(prediction.point)[0],
                    "quantile_levels": (
                        None
                        if prediction.quantile_levels is None
                        else _array_blob(prediction.quantile_levels)[0]
                    ),
                    "quantiles": (
                        None
                        if prediction.quantiles is None
                        else _array_blob(prediction.quantiles)[0]
                    ),
                    "timestamps": _array_blob(
                        prediction.timestamps.astype(np.int64, copy=False)
                    )[0],
                },
                "schema": _CELL_RESULT_SCHEMA,
                "target_sha256": target.sha256,
            }
        )
        atlas_sha256 = sha256_json(
            {
                "citations": _citation_record(cached.result),
                "schema": _ATLAS_CITATIONS_SCHEMA,
            }
        )
        return _PendingEvidence(
            target=target,
            coverage_gap=coverage - self.nominal_coverage,
            pinball=score.mae if score.pinball is None else score.pinball,
            mae=score.mae,
            joint_miss=joint_miss,
            normalized_error=normalized_error,
            route_w1=cached.result.route_w1,
            observed_values=score.observed_values,
            result_sha256=result_sha256,
            atlas_sha256=atlas_sha256,
            started_at=cached.started_at,
            completed_at=cached.completed_at,
            shadow_identity_sha256=_shadow_identity(target, cached.result),
        )

    @staticmethod
    def _materialize_evidence(
        pending: _PendingEvidence,
        shadow: ShadowProbeUpdate,
        work_cost: float,
    ) -> ProbeEvidence:
        return ProbeEvidence(
            target_sha256=pending.target.sha256,
            coverage_gap=pending.coverage_gap,
            pinball=pending.pinball,
            mae=pending.mae,
            joint_miss=pending.joint_miss,
            o1_surprise=shadow.surprise,
            learning_progress=shadow.learning_progress,
            route_w1=pending.route_w1,
            observed_values=pending.observed_values,
            work_cost=work_cost,
            result_sha256=pending.result_sha256,
            atlas_sha256=pending.atlas_sha256,
            started_at=pending.started_at,
            completed_at=pending.completed_at,
        )

    def _execute_group(
        self,
        targets: Sequence[ForecastProbeTarget],
    ) -> CoalescedProbeBatch:
        ordered = tuple(sorted(targets, key=lambda item: item.sha256))
        if not ordered:
            raise ValueError("execute group must contain at least one target")
        new_targets = tuple(
            target for target in ordered if target.sha256 not in self._evidence
        )
        key_sha256 = execution_key(ordered[0])
        if any(execution_key(target) != key_sha256 for target in ordered):
            raise CartographyExecutionContractError(
                "execute group contains multiple physical execution keys"
            )
        if not new_targets:
            cached_replay = self._forwards.get(key_sha256)
            result_sha256 = (
                cached_replay.result_sha256
                if cached_replay is not None
                else sha256_json(
                    {
                        "cell_results": [
                            self._evidence[target.sha256].result_sha256
                            for target in ordered
                        ],
                        "schema": "o1fm.cartography-replay-batch/v1",
                    }
                )
            )
            replay = CoalescedProbeBatch(
                execution_key_sha256=key_sha256,
                execution_result_sha256=result_sha256,
                target_sha256s=tuple(target.sha256 for target in ordered),
                evidence={
                    target.sha256: self._evidence[target.sha256] for target in ordered
                },
                work_charges={target.sha256: 0.0 for target in ordered},
                byte_charges={target.sha256: 0 for target in ordered},
                physical_work=0.0,
                payload_bytes=0,
                ledger_bytes=0,
                forward_count=0,
            )
            self._batches.append(replay)
            return replay
        windows = {target.sha256: self._resolve(target) for target in ordered}
        cached = self._forwards.get(key_sha256)
        required_horizon = max(target.horizon for target in ordered)
        fresh_forward = (
            cached is None
            or cached.result.prediction.horizon < required_horizon
            or any(
                not _same_model_prefix(
                    cached.window,
                    windows[target.sha256],
                    target.horizon,
                )
                for target in ordered
            )
        )
        cache_candidate: _CachedForward | None = None
        if fresh_forward:
            candidate = self._physical_forward(key_sha256, ordered, windows)
            work_charges = _allocate_work(candidate.work_cost, new_targets)
            work_charges.update(
                {
                    target.sha256: 0.0
                    for target in ordered
                    if target.sha256 not in work_charges
                }
            )
            byte_charges, byte_shortfall = _allocate_bytes(
                candidate.payload_bytes,
                new_targets,
            )
            byte_charges.update(
                {
                    target.sha256: 0
                    for target in ordered
                    if target.sha256 not in byte_charges
                }
            )
            if self.strict_byte_reservations and byte_shortfall:
                raise CartographyExecutionBudgetError(
                    "execution payload exceeds coalesced expected_bytes"
                )
            cached = candidate
            cache_candidate = candidate
            physical_work = cached.work_cost
            payload_bytes = cached.payload_bytes
            ledger_bytes = sum(target.expected_bytes for target in new_targets)
            forward_count = cached.forward_count
        else:
            assert cached is not None
            work_charges = {target.sha256: 0.0 for target in ordered}
            byte_charges = {target.sha256: 0 for target in ordered}
            byte_shortfall = 0
            physical_work = 0.0
            payload_bytes = 0
            ledger_bytes = sum(target.expected_bytes for target in new_targets)
            forward_count = 0
        pending = tuple(
            self._score_cell(target, windows[target.sha256], cached)
            for target in new_targets
        )

        def materialize(
            updates: tuple[ShadowProbeUpdate, ...],
        ) -> tuple[dict[str, ProbeEvidence], CoalescedProbeBatch]:
            if len(updates) != len(pending):
                raise AssertionError("shadow transaction cardinality changed")
            created = {
                item.target.sha256: self._materialize_evidence(
                    item,
                    update,
                    work_charges[item.target.sha256],
                )
                for item, update in zip(pending, updates, strict=True)
            }
            combined = {
                target.sha256: (
                    self._evidence[target.sha256]
                    if target.sha256 in self._evidence
                    else created[target.sha256]
                )
                for target in ordered
            }
            prepared_batch = CoalescedProbeBatch(
                execution_key_sha256=key_sha256,
                execution_result_sha256=cached.result_sha256,
                target_sha256s=tuple(target.sha256 for target in ordered),
                evidence=combined,
                work_charges=work_charges,
                byte_charges=byte_charges,
                physical_work=physical_work,
                payload_bytes=payload_bytes,
                ledger_bytes=ledger_bytes,
                forward_count=forward_count,
                reservation_shortfall_bytes=byte_shortfall,
            )
            return created, prepared_batch

        new_evidence, batch = self.shadow._transaction(
            tuple(
                (item.shadow_identity_sha256, item.normalized_error) for item in pending
            ),
            materialize,
        )
        if cache_candidate is not None:
            self._forwards[key_sha256] = cache_candidate
        self._evidence.update(new_evidence)
        self._batches.append(batch)
        return batch

    def execute_many(
        self,
        targets: Sequence[ForecastProbeTarget],
    ) -> tuple[CoalescedProbeBatch, ...]:
        """Synchronously execute every key, coalescing compatible cells."""

        if isinstance(targets, (str, bytes)) or not isinstance(targets, Sequence):
            raise TypeError("targets must be a sequence")
        selected = tuple(targets)
        if not selected:
            return ()
        if not all(isinstance(target, ForecastProbeTarget) for target in selected):
            raise TypeError("targets must contain ForecastProbeTarget instances")
        unique = {target.sha256: target for target in selected}
        if len(unique) != len(selected):
            selected = tuple(unique[key] for key in sorted(unique))
        with self._lock:
            self.register(selected)
            groups: dict[str, list[ForecastProbeTarget]] = {}
            for target in selected:
                groups.setdefault(execution_key(target), []).append(target)
            return tuple(self._execute_group(groups[key]) for key in sorted(groups))

    def prime(
        self,
        targets: Sequence[ForecastProbeTarget],
    ) -> tuple[CoalescedProbeBatch, ...]:
        """Alias used before handing this callable to ``background_run``."""

        return self.execute_many(targets)

    def __call__(self, target: ForecastProbeTarget) -> ProbeEvidence:
        if not isinstance(target, ForecastProbeTarget):
            raise TypeError("target must be ForecastProbeTarget")
        with self._lock:
            previous = self._evidence.get(target.sha256)
            if previous is not None:
                return previous
            self.register((target,))
            key_sha256 = execution_key(target)
            anchor = self._resolve(target)
            group: list[ForecastProbeTarget] = [target]
            group_windows: dict[str, ProbeWindow] = {target.sha256: anchor}
            for candidate in sorted(
                self._registered.values(),
                key=lambda item: item.sha256,
            ):
                if (
                    candidate.sha256 == target.sha256
                    or execution_key(candidate) != key_sha256
                    or candidate.sha256 in self._evidence
                ):
                    continue
                try:
                    candidate_window = self._resolve(candidate)
                except (
                    CartographyExecutionError,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    # A malformed or not-yet-realized peer remains pending and
                    # cannot poison the requested serial cartography step.
                    continue
                compatible = all(
                    _same_model_prefix(
                        existing_window,
                        candidate_window,
                        min(existing.horizon, candidate.horizon),
                    )
                    for existing in group
                    for existing_window in (group_windows[existing.sha256],)
                )
                if compatible:
                    group.append(candidate)
                    group_windows[candidate.sha256] = candidate_window
            batch = self._execute_group(group)
            return batch.evidence_for(target)


__all__ = [
    "BranchProbeCallable",
    "CartographyExecutionBridge",
    "CartographyExecutionBudgetError",
    "CartographyExecutionContractError",
    "CartographyExecutionError",
    "CartographyProbeNotReadyError",
    "CoalescedProbeBatch",
    "MeasuredBranchExecution",
    "ProbeWindow",
    "ShadowProbeLedger",
    "ShadowProbeUpdate",
    "WindowResolver",
    "execution_key",
]
