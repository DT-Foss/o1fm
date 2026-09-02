"""Append-only ``.causal`` citations for O1FM forecast execution.

The adapter deliberately depends on a four-method structural protocol.  The
IMMER ``LiveStore`` implementation is imported only by :meth:`from_immer`, so
O1FM remains independently importable and installable.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .identity import (
    IdentityError,
    canonical_json_bytes,
    require_sha256,
    sha256_bytes,
    sha256_json,
)

FORECAST_ATLAS_SCHEMA = "o1fm.forecast-atlas.forecast/v1"
SETTLEMENT_ATLAS_SCHEMA = "o1fm.forecast-atlas.settlement/v2"
TRANSPORT_ATLAS_SCHEMA = "o1fm.forecast-atlas.transport/v1"

_KINDS = frozenset({"forecast", "settlement", "transport"})
_ACTIONS = frozenset({"foundation", "local", "crystal", "seasonal"})
_ZERO_SHA256 = "0" * 64
_MAX_RECORDS = 1_000_000


class ForecastAtlasError(RuntimeError):
    """The Atlas store or an append receipt violates its contract."""


class ForecastAtlasValidationError(ForecastAtlasError, ValueError):
    """An object cannot be represented by the Forecast Atlas schema."""


class ForecastAtlasIntegrityError(ForecastAtlasError):
    """Persisted Atlas evidence does not match its exact citation."""


@runtime_checkable
class SegmentStore(Protocol):
    """Minimal content-addressed segment contract used by ``ForecastAtlas``."""

    def append_segment(self, records: Iterable[Mapping[str, Any]]) -> str: ...

    def revision(self) -> tuple[int, str]: ...

    def record(self, sha: str, index: int) -> Mapping[str, Any]: ...

    def verify(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class AtlasReceipt:
    """Exact direct citation for one atomically activated Atlas segment."""

    segment_sha256: str
    revision_sequence: int
    revision_sha256: str
    record_count: int
    kind: str

    def __post_init__(self) -> None:
        try:
            require_sha256(self.segment_sha256, field="segment_sha256")
            require_sha256(self.revision_sha256, field="revision_sha256")
        except IdentityError as exc:
            raise ForecastAtlasValidationError(str(exc)) from exc
        if (
            isinstance(self.revision_sequence, bool)
            or not isinstance(self.revision_sequence, int)
            or self.revision_sequence < 0
        ):
            raise ForecastAtlasValidationError(
                "revision_sequence must be a non-negative integer"
            )
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or not 1 <= self.record_count <= _MAX_RECORDS
        ):
            raise ForecastAtlasValidationError(
                "record_count must be a positive bounded integer"
            )
        if self.kind not in _KINDS:
            raise ForecastAtlasValidationError(f"kind must be one of {sorted(_KINDS)}")


def _field(value: object, name: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise ForecastAtlasValidationError(f"object requires {name}") from exc


def _text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ForecastAtlasValidationError(f"{field} must be canonical non-empty text")
    try:
        canonical_json_bytes(value)
    except IdentityError as exc:
        raise ForecastAtlasValidationError(f"{field} is not canonical text") from exc
    return value


def _finite(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ForecastAtlasValidationError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ForecastAtlasValidationError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


def _integer(value: object, *, field: str, nonnegative: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ForecastAtlasValidationError(f"{field} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ForecastAtlasValidationError(f"{field} must be non-negative")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ForecastAtlasValidationError(f"{field} must be boolean")
    return bool(value)


def _canonical_copy(value: object) -> object:
    try:
        return json.loads(canonical_json_bytes(value))
    except (IdentityError, json.JSONDecodeError) as exc:
        raise ForecastAtlasValidationError("Atlas data is not canonical JSON") from exc


def _document(value: object, *, field: str) -> dict[str, object]:
    method = getattr(value, "to_document", None)
    if method is None:
        method = getattr(value, "to_record", None)
    if not callable(method):
        raise ForecastAtlasValidationError(
            f"{field} must provide to_document() or to_record()"
        )
    result = _canonical_copy(method())
    if not isinstance(result, dict) or not result:
        raise ForecastAtlasValidationError(f"{field} document must be non-empty")
    return result


def _timestamps(value: object) -> dict[str, object]:
    try:
        timestamps = np.asarray(value, dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ForecastAtlasValidationError("timestamps must be valid") from exc
    if timestamps.ndim != 1 or timestamps.size < 1 or bool(np.isnat(timestamps).any()):
        raise ForecastAtlasValidationError("timestamps must be a non-empty vector")
    ticks = timestamps.astype(np.int64, copy=False)
    if ticks.size > 1 and bool((np.diff(ticks) <= 0).any()):
        raise ForecastAtlasValidationError("timestamps must be strictly increasing")
    values = [int(item) for item in ticks]
    return {
        "count": len(values),
        "end_ns": values[-1],
        "sha256": sha256_json(values),
        "start_ns": values[0],
        "values_ns": values,
    }


def _array_identity(value: object, *, field: str) -> dict[str, object]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or raw.ndim < 1 or np.iscomplexobj(raw):
        raise ForecastAtlasValidationError(f"{field} must be a real numeric array")
    if not bool(np.isfinite(raw).all()):
        raise ForecastAtlasValidationError(f"{field} must contain finite values")
    dtype = raw.dtype.newbyteorder("<")
    array = np.ascontiguousarray(raw, dtype=dtype)
    return {
        "data_sha256": sha256_bytes(array.tobytes(order="C")),
        "dtype": array.dtype.str,
        "shape": [int(axis) for axis in array.shape],
    }


def _prediction_identity(prediction: object) -> tuple[str, dict[str, object]]:
    series_id = _text(_field(prediction, "series_id"), field="prediction.series_id")
    timestamp_record = _timestamps(_field(prediction, "timestamps"))
    point = _array_identity(_field(prediction, "point"), field="prediction.point")
    if len(point["shape"]) != 2 or any(axis <= 0 for axis in point["shape"]):
        raise ForecastAtlasValidationError(
            "prediction.point must be a non-empty horizon-by-target matrix"
        )
    quantiles = getattr(prediction, "quantiles", None)
    levels = getattr(prediction, "quantile_levels", None)
    if (quantiles is None) != (levels is None):
        raise ForecastAtlasValidationError(
            "prediction quantiles and levels must be supplied together"
        )
    identity: dict[str, object] = {
        "point": point,
        "series_id": series_id,
        "timestamps_sha256": timestamp_record["sha256"],
    }
    metadata: dict[str, object] = {
        "horizon": timestamp_record["count"],
        "quantile_levels": None,
        "target_count": point["shape"][1],
    }
    if quantiles is not None:
        identity["quantiles"] = _array_identity(
            quantiles,
            field="prediction.quantiles",
        )
        quantile_shape = identity["quantiles"]["shape"]
        level_array = np.asarray(levels, dtype=np.float64)
        if (
            level_array.ndim != 1
            or level_array.size < 1
            or not bool(np.isfinite(level_array).all())
            or bool((level_array <= 0.0).any())
            or bool((level_array >= 1.0).any())
            or bool((np.diff(level_array) <= 0.0).any())
            or quantile_shape[:2] != point["shape"]
            or quantile_shape[2:] != [int(level_array.size)]
        ):
            raise ForecastAtlasValidationError("quantile levels are invalid")
        level_values = [_finite(value, field="quantile_level") for value in level_array]
        identity["quantile_levels"] = level_values
        metadata["quantile_levels"] = level_values
    digest = sha256_json(identity)
    metadata["sha256"] = digest
    return digest, metadata


def _profile(value: object) -> tuple[dict[str, object], str]:
    document = _document(value, field="profile")
    required = {
        "feature_schema",
        "graph_revision",
        "horizon",
        "model_identity",
        "regime",
        "target_count",
    }
    if set(document) != required:
        raise ForecastAtlasValidationError("profile fields are invalid")
    for field in ("feature_schema", "graph_revision", "model_identity"):
        try:
            require_sha256(document[field], field=f"profile.{field}")
        except IdentityError as exc:
            raise ForecastAtlasValidationError(str(exc)) from exc
    _text(document["regime"], field="profile.regime")
    if _integer(document["horizon"], field="profile.horizon") == 0:
        raise ForecastAtlasValidationError("profile.horizon must be positive")
    if _integer(document["target_count"], field="profile.target_count") == 0:
        raise ForecastAtlasValidationError("profile.target_count must be positive")
    return document, sha256_json(document)


def _edge_topology(edge: object) -> dict[str, object]:
    kind = _text(_field(edge, "kind"), field="edge.kind")
    if kind not in {"observational", "intervention"}:
        raise ForecastAtlasValidationError("edge.kind is invalid")
    revision = _field(edge, "revision_sha256")
    try:
        require_sha256(revision, field="edge.revision_sha256")
    except IdentityError as exc:
        raise ForecastAtlasValidationError(str(exc)) from exc
    return {
        "kind": kind,
        "lag": _integer(_field(edge, "lag"), field="edge.lag"),
        "revision_sha256": revision,
        "source": _text(_field(edge, "source"), field="edge.source"),
        "target": _text(_field(edge, "target"), field="edge.target"),
    }


def _route(value: object) -> tuple[dict[str, object], dict[str, object]]:
    signature = _field(value, "signature_sha256")
    try:
        require_sha256(signature, field="route.signature_sha256")
    except IdentityError as exc:
        raise ForecastAtlasValidationError(str(exc)) from exc
    past = tuple(
        _integer(item, field="route.past_index")
        for item in _field(value, "past_indices")
    )
    known = tuple(
        _integer(item, field="route.known_index")
        for item in _field(value, "known_indices")
    )
    if past != tuple(sorted(set(past))) or known != tuple(sorted(set(known))):
        raise ForecastAtlasValidationError("route indices must be sorted and unique")
    topology = tuple(
        sorted(
            (_edge_topology(edge) for edge in _field(value, "edges")),
            key=lambda edge: (
                edge["source"],
                edge["target"],
                edge["lag"],
                edge["kind"],
            ),
        )
    )
    revisions = sorted({str(edge["revision_sha256"]) for edge in topology})
    if len(revisions) > 1:
        raise ForecastAtlasValidationError("route edges do not share one graph head")
    graph = {
        "head_sha256": revisions[0] if revisions else _ZERO_SHA256,
        "topology": list(topology),
        "topology_sha256": signature,
    }
    route = {
        "known_indices": list(known),
        "past_indices": list(past),
        "signature_sha256": signature,
        "transport_w1": _finite(
            getattr(value, "transport_w1", 0.0),
            field="route.transport_w1",
            nonnegative=True,
        ),
    }
    expected_signature = sha256_json(
        {
            "edges": [
                {
                    "kind": edge["kind"],
                    "lag": edge["lag"],
                    "source": edge["source"],
                    "target": edge["target"],
                }
                for edge in topology
            ],
            "known": list(known),
            "past": list(past),
            "schema": "o1fm.routed-feature-topology/v1",
        }
    )
    if expected_signature != signature:
        raise ForecastAtlasValidationError(
            "route signature does not authenticate its topology"
        )
    return route, graph


def _calibration(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    key = _document(_field(value, "key"), field="calibration.key")
    base_sha, _ = _prediction_identity(_field(value, "base_prediction"))
    calibrated_sha, _ = _prediction_identity(_field(value, "prediction"))
    record = {
        "base_prediction_sha256": base_sha,
        "joint_alpha": _finite(_field(value, "joint_alpha"), field="joint_alpha"),
        "key": key,
        "lower_alpha": _finite(_field(value, "lower_alpha"), field="lower_alpha"),
        "prediction_sha256": calibrated_sha,
        "prior_settlements": _integer(
            _field(value, "prior_settlements"),
            field="prior_settlements",
        ),
        "upper_alpha": _finite(_field(value, "upper_alpha"), field="upper_alpha"),
    }
    return {"identity_sha256": sha256_json(record), **record}


def _ticket_context(ticket: object) -> dict[str, object]:
    ticket_id = _field(ticket, "ticket_id")
    try:
        require_sha256(ticket_id, field="ticket_id")
    except IdentityError as exc:
        raise ForecastAtlasValidationError(str(exc)) from exc
    profile, profile_sha = _profile(_field(ticket, "profile"))
    route, graph = _route(_field(ticket, "route"))
    if route["signature_sha256"] != profile["graph_revision"]:
        raise ForecastAtlasValidationError("profile and route revisions differ")
    regime = _text(_field(ticket, "regime"), field="ticket.regime")
    if regime != profile["regime"]:
        raise ForecastAtlasValidationError("ticket and profile regimes differ")
    action = _text(_field(ticket, "action"), field="ticket.action")
    if action not in _ACTIONS:
        raise ForecastAtlasValidationError(
            f"ticket.action must be one of {sorted(_ACTIONS)}"
        )
    prediction_sha, prediction = _prediction_identity(_field(ticket, "prediction"))
    base_sha, _base = _prediction_identity(_field(ticket, "base_prediction"))
    inputs = _field(ticket, "inputs")
    series_id = _text(_field(inputs, "series_id"), field="inputs.series_id")
    if series_id != _field(_field(ticket, "prediction"), "series_id"):
        raise ForecastAtlasValidationError("input and prediction series differ")
    timestamps = _timestamps(_field(_field(ticket, "prediction"), "timestamps"))
    past_timestamps = _timestamps(_field(inputs, "past_timestamps"))
    if (
        prediction["horizon"] != profile["horizon"]
        or prediction["target_count"] != profile["target_count"]
    ):
        raise ForecastAtlasValidationError("prediction and profile shapes differ")
    if past_timestamps["end_ns"] >= timestamps["start_ns"]:
        raise ForecastAtlasValidationError("forecast horizon must follow its origin")
    timestamps["forecast_origin_ns"] = past_timestamps["end_ns"]
    return {
        "action": action,
        "base_prediction_sha256": base_sha,
        "calibration": _calibration(getattr(ticket, "calibration", None)),
        "graph": graph,
        "model_identity": profile["model_identity"],
        "prediction": prediction,
        "prediction_sha256": prediction_sha,
        "profile": profile,
        "profile_sha256": profile_sha,
        "regime": regime,
        "route": route,
        "series_id": series_id,
        "ticket_id": ticket_id,
        "timestamps": timestamps,
    }


def _score(value: object) -> dict[str, object]:
    def optional(name: str) -> float | None:
        selected = _field(value, name)
        return None if selected is None else _finite(selected, field=f"score.{name}")

    return {
        "interval_coverage": optional("interval_coverage"),
        "interval_width": optional("interval_width"),
        "mae": _finite(_field(value, "mae"), field="score.mae", nonnegative=True),
        "observed_values": _integer(
            _field(value, "observed_values"),
            field="score.observed_values",
        ),
        "pinball": optional("pinball"),
        "rmse": _finite(_field(value, "rmse"), field="score.rmse", nonnegative=True),
    }


def _o1_update(value: object) -> dict[str, object]:
    document = _document(value, field="o1_update")
    observation_sha = _field(value, "observation_sha256")
    try:
        require_sha256(observation_sha, field="o1.observation_sha256")
    except IdentityError as exc:
        raise ForecastAtlasValidationError(str(exc)) from exc
    return {
        "cusum": _finite(_field(value, "cusum"), field="o1.cusum", nonnegative=True),
        "identity_sha256": sha256_json(document),
        "observation_sha256": observation_sha,
        "regime_changed": _boolean(
            _field(value, "regime_changed"),
            field="o1.regime_changed",
        ),
        "regime_generation": _integer(
            _field(value, "regime_generation"),
            field="o1.regime_generation",
        ),
        "sequence": _integer(_field(value, "sequence"), field="o1.sequence"),
        "surprise": _finite(
            _field(value, "surprise"),
            field="o1.surprise",
            nonnegative=True,
        ),
    }


def _conformal_update(value: object | None) -> dict[str, object] | None:
    if value is None:
        return None
    content_sha = _field(value, "content_sha256")
    try:
        require_sha256(content_sha, field="conformal.content_sha256")
    except IdentityError as exc:
        raise ForecastAtlasValidationError(str(exc)) from exc
    joint_hit = _field(value, "joint_hit")
    if joint_hit is not None:
        joint_hit = _boolean(joint_hit, field="conformal.joint_hit")
    return {
        "content_sha256": content_sha,
        "joint_alpha": _finite(_field(value, "joint_alpha"), field="joint_alpha"),
        "joint_hit": joint_hit,
        "lower_alpha": _finite(_field(value, "lower_alpha"), field="lower_alpha"),
        "observed_values": _integer(
            _field(value, "observed_values"),
            field="conformal.observed_values",
        ),
        "upper_alpha": _finite(_field(value, "upper_alpha"), field="upper_alpha"),
    }


def _segment_sha(records: tuple[dict[str, object], ...]) -> str:
    return sha256_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )


class ForecastAtlas:
    """Persist deterministic forecast decisions as directly citable segments."""

    def __init__(self, store: SegmentStore) -> None:
        if not isinstance(store, SegmentStore):
            raise TypeError("store must implement SegmentStore")
        self.store = store

    @classmethod
    def from_immer(cls, directory: str | Path) -> ForecastAtlas:
        """Mount IMMER's ``LiveStore`` without making it an O1FM dependency."""

        try:
            from immer.knowledge.livecausal import LiveStore
        except ImportError as exc:
            raise ImportError(
                "ForecastAtlas.from_immer() requires IMMER with "
                "immer.knowledge.livecausal.LiveStore available"
            ) from exc
        return cls(LiveStore(directory))

    @staticmethod
    def _forecast_record(ticket: object) -> dict[str, object]:
        context = _ticket_context(ticket)
        costs = {
            "forced_foundation_refresh": _boolean(
                _field(ticket, "forced_foundation_refresh"),
                field="forced_foundation_refresh",
            ),
            "forward_cost": _finite(
                _field(ticket, "forward_cost"),
                field="forward_cost",
                nonnegative=True,
            ),
            "foundation_forwards": _integer(
                _field(ticket, "foundation_forwards"),
                field="foundation_forwards",
            ),
            "local_forwards": _integer(
                _field(ticket, "local_forwards"),
                field="local_forwards",
            ),
        }
        return {
            **context,
            "costs": costs,
            "kind": "forecast",
            "outcome_key": f"o1fm:prediction:{context['prediction_sha256']}",
            "schema": FORECAST_ATLAS_SCHEMA,
            "trigger_key": (
                f"o1fm:forecast-origin:{context['series_id']}:"
                f"{context['timestamps']['forecast_origin_ns']}:"
                f"{context['profile_sha256']}"
            ),
        }

    @staticmethod
    def _settlement_record(ticket: object, settlement: object) -> dict[str, object]:
        context = _ticket_context(ticket)
        if _field(settlement, "ticket_id") != context["ticket_id"]:
            raise ForecastAtlasValidationError("settlement ticket identity differs")
        if _field(settlement, "action") != context["action"]:
            raise ForecastAtlasValidationError("settlement action differs")
        scores = {
            "base": _score(_field(settlement, "base_score")),
            "selected": _score(_field(settlement, "score")),
        }
        o1 = _o1_update(_field(settlement, "o1_update"))
        conformal = _conformal_update(getattr(settlement, "calibration_update", None))
        costs = {
            "forced_foundation_refresh": _boolean(
                _field(settlement, "forced_foundation_refresh"),
                field="forced_foundation_refresh",
            ),
            "forward_cost": _finite(
                _field(settlement, "forward_cost"),
                field="forward_cost",
                nonnegative=True,
            ),
            "foundation_forwards": _integer(
                _field(settlement, "foundation_forwards"),
                field="foundation_forwards",
            ),
            "local_forwards": _integer(
                _field(settlement, "local_forwards"),
                field="local_forwards",
            ),
            "saved_foundation_forwards": _integer(
                _field(settlement, "saved_foundation_forwards"),
                field="saved_foundation_forwards",
            ),
        }
        fragment_sha256 = getattr(settlement, "fragment_sha256", _ZERO_SHA256)
        try:
            fragment_sha256 = require_sha256(
                fragment_sha256,
                field="fragment_sha256",
            )
        except IdentityError as exc:
            raise ForecastAtlasValidationError(str(exc)) from exc
        observed_values = _integer(
            getattr(
                settlement,
                "observed_values",
                _field(_field(settlement, "score"), "observed_values"),
            ),
            field="settlement.observed_values",
        )
        settlement_identity = sha256_json(
            {
                "conformal_update": conformal,
                "costs": costs,
                "fragment_sha256": fragment_sha256,
                "observed_values": observed_values,
                "o1_update": o1,
                "scores": scores,
                "ticket_id": context["ticket_id"],
            }
        )
        joint_covered = _field(settlement, "joint_covered")
        if joint_covered is not None:
            joint_covered = _boolean(joint_covered, field="joint_covered")
        return {
            **context,
            "conformal_update": conformal,
            "costs": costs,
            "crystal_ready": _boolean(
                _field(settlement, "crystal_ready"),
                field="crystal_ready",
            ),
            "fragment_sha256": fragment_sha256,
            "joint_covered": joint_covered,
            "kind": "settlement",
            "observed_values": observed_values,
            "o1_update": o1,
            "outcome_key": f"o1fm:settlement:{settlement_identity}",
            "schema": SETTLEMENT_ATLAS_SCHEMA,
            "scores": scores,
            "settlement_sha256": settlement_identity,
            "trigger_key": f"o1fm:prediction:{context['prediction_sha256']}",
        }

    @staticmethod
    def _transport_record(ticket: object) -> dict[str, object]:
        context = _ticket_context(ticket)
        transport_identity = sha256_json(
            {
                "graph": context["graph"],
                "profile_sha256": context["profile_sha256"],
                "route": context["route"],
                "ticket_id": context["ticket_id"],
            }
        )
        return {
            **context,
            "kind": "transport",
            "outcome_key": f"o1fm:transport:{transport_identity}",
            "schema": TRANSPORT_ATLAS_SCHEMA,
            "transport_sha256": transport_identity,
            "trigger_key": f"o1fm:profile:{context['profile_sha256']}",
        }

    def _append(
        self, records: tuple[dict[str, object], ...], *, kind: str
    ) -> AtlasReceipt:
        if not records or kind not in _KINDS:
            raise ForecastAtlasValidationError("Atlas append is invalid")
        normalized = tuple(_canonical_copy(record) for record in records)
        assert all(isinstance(record, dict) for record in normalized)
        canonical_records = tuple(
            record for record in normalized if isinstance(record, dict)
        )
        expected_sha = _segment_sha(canonical_records)
        before_sequence, before_sha = self._revision()
        actual_sha = self.store.append_segment(canonical_records)
        try:
            require_sha256(actual_sha, field="segment_sha256")
        except IdentityError as exc:
            raise ForecastAtlasIntegrityError(str(exc)) from exc
        if actual_sha != expected_sha:
            raise ForecastAtlasIntegrityError(
                "store returned the wrong segment identity"
            )
        after_sequence, after_sha = self._revision()
        if after_sequence < before_sequence:
            raise ForecastAtlasIntegrityError("store revision moved backwards")
        if after_sequence == before_sequence and after_sha != before_sha:
            raise ForecastAtlasIntegrityError(
                "unchanged revision sequence changed hash"
            )
        if after_sequence > before_sequence and after_sha == before_sha:
            raise ForecastAtlasIntegrityError("advanced revision reused its prior hash")
        for index, expected in enumerate(canonical_records):
            try:
                observed = self.store.record(actual_sha, index)
            except Exception as exc:
                raise ForecastAtlasIntegrityError(
                    "appended segment is not directly citable"
                ) from exc
            try:
                same = canonical_json_bytes(observed) == canonical_json_bytes(expected)
            except IdentityError as exc:
                raise ForecastAtlasIntegrityError(
                    "cited record is not canonical"
                ) from exc
            if not same:
                raise ForecastAtlasIntegrityError("cited record differs from append")
        if not self.store.verify():
            raise ForecastAtlasIntegrityError("segment store verification failed")
        return AtlasReceipt(
            segment_sha256=actual_sha,
            revision_sequence=after_sequence,
            revision_sha256=after_sha,
            record_count=len(canonical_records),
            kind=kind,
        )

    def _revision(self) -> tuple[int, str]:
        revision = self.store.revision()
        if (
            not isinstance(revision, tuple)
            or len(revision) != 2
            or isinstance(revision[0], bool)
            or not isinstance(revision[0], int)
            or revision[0] < 0
        ):
            raise ForecastAtlasIntegrityError("store revision is invalid")
        try:
            require_sha256(revision[1], field="revision_sha256")
        except IdentityError as exc:
            raise ForecastAtlasIntegrityError(str(exc)) from exc
        return revision

    def append_forecast(self, ticket: object) -> AtlasReceipt:
        return self._append((self._forecast_record(ticket),), kind="forecast")

    def append_settlement(self, ticket: object, settlement: object) -> AtlasReceipt:
        return self._append(
            (self._settlement_record(ticket, settlement),),
            kind="settlement",
        )

    def append_transport(self, ticket: object) -> AtlasReceipt:
        return self._append((self._transport_record(ticket),), kind="transport")

    def record(self, receipt: AtlasReceipt, index: int = 0) -> Mapping[str, Any]:
        if not isinstance(receipt, AtlasReceipt):
            raise TypeError("receipt must be AtlasReceipt")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < receipt.record_count
        ):
            raise ForecastAtlasValidationError("record index is outside the receipt")
        try:
            record = self.store.record(receipt.segment_sha256, index)
            canonical = _canonical_copy(record)
        except Exception as exc:
            raise ForecastAtlasIntegrityError("cannot resolve Atlas citation") from exc
        if not isinstance(canonical, dict) or canonical.get("kind") != receipt.kind:
            raise ForecastAtlasIntegrityError("cited record kind differs from receipt")
        return canonical

    def verify(self, receipt: AtlasReceipt | None = None) -> bool:
        if not self.store.verify():
            return False
        if receipt is None:
            return True
        if not isinstance(receipt, AtlasReceipt):
            raise TypeError("receipt must be AtlasReceipt or None")
        try:
            records = tuple(
                self.store.record(receipt.segment_sha256, index)
                for index in range(receipt.record_count)
            )
            normalized = tuple(_canonical_copy(record) for record in records)
            if not all(
                isinstance(record, dict) and record.get("kind") == receipt.kind
                for record in normalized
            ):
                return False
            return (
                _segment_sha(
                    tuple(record for record in normalized if isinstance(record, dict))
                )
                == receipt.segment_sha256
            )
        except Exception:
            return False


__all__ = [
    "FORECAST_ATLAS_SCHEMA",
    "SETTLEMENT_ATLAS_SCHEMA",
    "TRANSPORT_ATLAS_SCHEMA",
    "AtlasReceipt",
    "ForecastAtlas",
    "ForecastAtlasError",
    "ForecastAtlasIntegrityError",
    "ForecastAtlasValidationError",
    "SegmentStore",
]
