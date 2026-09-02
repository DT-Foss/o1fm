"""Persistent directed conformal calibration driven by O1 forecast errors.

The field keeps lower-tail, upper-tail, and trajectory-wise error processes
separate.  Exact profiles remain action, route, and regime specific while a
second pooled level carries evidence into newly selected runtime profiles.
Only settled targets enter the state; :meth:`apply` is a prior-only operation.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .contracts import ForecastPrediction, ForecastTarget
from .identity import canonical_json_bytes, sha256_json

try:  # POSIX is the production target; the in-process lock stays portable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


CONFORMAL_FIELD_SCHEMA = "o1fm.conformal-field/v1"
CONFORMAL_CONFIG_SCHEMA = "o1fm.conformal-config/v1"
CONFORMAL_KEY_SCHEMA = "o1fm.conformal-key/v1"
_STATE_FILE = "conformal-field.json"
_LOCK_FILE = ".conformal-field.lock"
_ACTIONS = frozenset({"foundation", "crystal", "seasonal"})
_MAX_TEXT_BYTES = 4096
_MAX_AXIS = 4096
_MAX_CELLS = 65_536
_MAX_HISTORY = 100_000
_MAX_OBSERVATION_IDS = 1_000_000
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}

Float32Matrix = NDArray[np.float32]


class ConformalFieldError(RuntimeError):
    """Persistent conformal state or settlement evidence is invalid."""


class ConformalIntegrityError(ConformalFieldError):
    """The persistent field violates its canonical state contract."""


class ConformalConflictError(ConformalFieldError):
    """An observation ID was replayed with different settlement evidence."""


def _finite(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return 0.0 if result == 0.0 else result


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= maximum
    ):
        raise ValueError(f"{field} must be a positive bounded integer")
    return int(value)


def _nonnegative_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) <= maximum
    ):
        raise ValueError(f"{field} must be a non-negative bounded integer")
    return int(value)


def _canonical_text(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > _MAX_TEXT_BYTES
    ):
        raise ValueError(f"{field} must be canonical bounded text")
    return value


def _readonly_float32(
    value: object, *, shape: tuple[int, ...], field: str
) -> NDArray[np.float32]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{field} cannot contain complex values")
    try:
        array = np.array(value, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError(f"{field} must be finite with shape {shape}")
    array[array == 0.0] = np.float32(0.0)
    array.setflags(write=False)
    return array


def _hex(value: float) -> str:
    return _finite(value, field="state float").hex()


def _from_hex(value: object, *, field: str, nonnegative: bool = False) -> float:
    if not isinstance(value, str) or len(value) > 32:
        raise ConformalIntegrityError(f"{field} is not a canonical float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ConformalIntegrityError(f"{field} is not a canonical float") from exc
    result = _finite(result, field=field, minimum=0.0 if nonnegative else None)
    if result.hex() != value:
        raise ConformalIntegrityError(f"{field} is not canonical hexadecimal float")
    return result


def _history_to_record(history: Sequence[float | None]) -> list[str | None]:
    return [None if value is None else _hex(value) for value in history]


def _history_from_record(value: object, *, field: str) -> list[float | None]:
    if not isinstance(value, list):
        raise ConformalIntegrityError(f"{field} must be a list")
    return [
        None if item is None else _from_hex(item, field=field, nonnegative=True)
        for item in value
    ]


@dataclass(frozen=True, slots=True)
class ConformalFieldConfig:
    """Pinned online-calibration policy for one persistent field."""

    target_coverage: float = 0.8
    base_gain: float = 0.02
    surprise_gain: float = 0.25
    surprise_cap: float = 4.0
    local_shrinkage: float = 4.0
    max_history: int = 512
    min_scale: float = 1e-4
    alpha_min: float = 0.005
    alpha_max: float = 0.5

    def __post_init__(self) -> None:
        coverage = _finite(
            self.target_coverage,
            field="target_coverage",
            minimum=0.0,
            maximum=1.0,
        )
        if coverage in (0.0, 1.0):
            raise ValueError("target_coverage must lie strictly inside (0, 1)")
        base_gain = _finite(self.base_gain, field="base_gain", minimum=0.0)
        if base_gain == 0.0:
            raise ValueError("base_gain must be positive")
        surprise_gain = _finite(
            self.surprise_gain,
            field="surprise_gain",
            minimum=0.0,
        )
        surprise_cap = _finite(
            self.surprise_cap,
            field="surprise_cap",
            minimum=0.0,
        )
        local_shrinkage = _finite(
            self.local_shrinkage,
            field="local_shrinkage",
            minimum=0.0,
        )
        max_history = _positive_int(
            self.max_history,
            field="max_history",
            maximum=_MAX_HISTORY,
        )
        min_scale = _finite(self.min_scale, field="min_scale", minimum=0.0)
        if min_scale == 0.0:
            raise ValueError("min_scale must be positive")
        alpha_min = _finite(
            self.alpha_min,
            field="alpha_min",
            minimum=0.0,
            maximum=1.0,
        )
        alpha_max = _finite(
            self.alpha_max,
            field="alpha_max",
            minimum=0.0,
            maximum=1.0,
        )
        if alpha_min == 0.0 or alpha_min >= alpha_max:
            raise ValueError("alpha bounds must satisfy 0 < alpha_min < alpha_max")
        object.__setattr__(self, "target_coverage", coverage)
        object.__setattr__(self, "base_gain", base_gain)
        object.__setattr__(self, "surprise_gain", surprise_gain)
        object.__setattr__(self, "surprise_cap", surprise_cap)
        object.__setattr__(self, "local_shrinkage", local_shrinkage)
        object.__setattr__(self, "max_history", max_history)
        object.__setattr__(self, "min_scale", min_scale)
        object.__setattr__(self, "alpha_min", alpha_min)
        object.__setattr__(self, "alpha_max", alpha_max)

    def to_record(self) -> dict[str, object]:
        return {
            "alpha_max": _hex(self.alpha_max),
            "alpha_min": _hex(self.alpha_min),
            "base_gain": _hex(self.base_gain),
            "local_shrinkage": _hex(self.local_shrinkage),
            "max_history": self.max_history,
            "min_scale": _hex(self.min_scale),
            "schema": CONFORMAL_CONFIG_SCHEMA,
            "surprise_cap": _hex(self.surprise_cap),
            "surprise_gain": _hex(self.surprise_gain),
            "target_coverage": _hex(self.target_coverage),
        }

    @classmethod
    def from_record(cls, value: object) -> ConformalFieldConfig:
        fields = {
            "alpha_max",
            "alpha_min",
            "base_gain",
            "local_shrinkage",
            "max_history",
            "min_scale",
            "schema",
            "surprise_cap",
            "surprise_gain",
            "target_coverage",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ConformalIntegrityError("conformal config fields are invalid")
        if value.get("schema") != CONFORMAL_CONFIG_SCHEMA:
            raise ConformalIntegrityError("conformal config schema is unsupported")
        try:
            result = cls(
                target_coverage=_from_hex(
                    value["target_coverage"], field="target_coverage"
                ),
                base_gain=_from_hex(value["base_gain"], field="base_gain"),
                surprise_gain=_from_hex(value["surprise_gain"], field="surprise_gain"),
                surprise_cap=_from_hex(value["surprise_cap"], field="surprise_cap"),
                local_shrinkage=_from_hex(
                    value["local_shrinkage"], field="local_shrinkage"
                ),
                max_history=value["max_history"],
                min_scale=_from_hex(value["min_scale"], field="min_scale"),
                alpha_min=_from_hex(value["alpha_min"], field="alpha_min"),
                alpha_max=_from_hex(value["alpha_max"], field="alpha_max"),
            )
        except (TypeError, ValueError) as exc:
            raise ConformalIntegrityError(
                "conformal config values are invalid"
            ) from exc
        if result.to_record() != dict(value):
            raise ConformalIntegrityError("conformal config is not canonical")
        return result


@dataclass(frozen=True, slots=True)
class ConformalFieldKey:
    """Exact execution identity of one calibration profile."""

    model_identity: str
    feature_schema: str
    graph_revision: str
    regime: str
    action: str
    horizon: int
    target_count: int
    lower_quantile: float
    upper_quantile: float

    def __post_init__(self) -> None:
        for field in ("model_identity", "feature_schema", "graph_revision", "regime"):
            object.__setattr__(
                self,
                field,
                _canonical_text(getattr(self, field), field=field),
            )
        action = _canonical_text(self.action, field="action")
        if action not in _ACTIONS:
            raise ValueError(f"action must be one of {sorted(_ACTIONS)}")
        horizon = _positive_int(self.horizon, field="horizon", maximum=_MAX_AXIS)
        target_count = _positive_int(
            self.target_count,
            field="target_count",
            maximum=_MAX_AXIS,
        )
        if horizon * target_count > _MAX_CELLS:
            raise ValueError("conformal profile exceeds the cell bound")
        lower = _finite(
            self.lower_quantile,
            field="lower_quantile",
            minimum=0.0,
            maximum=1.0,
        )
        upper = _finite(
            self.upper_quantile,
            field="upper_quantile",
            minimum=0.0,
            maximum=1.0,
        )
        if lower == 0.0 or upper == 1.0 or lower >= upper:
            raise ValueError("quantiles must satisfy 0 < lower < upper < 1")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "target_count", target_count)
        object.__setattr__(self, "lower_quantile", lower)
        object.__setattr__(self, "upper_quantile", upper)

    @property
    def shape(self) -> tuple[int, int]:
        return self.horizon, self.target_count

    def to_record(self) -> dict[str, object]:
        return {
            "action": self.action,
            "feature_schema": self.feature_schema,
            "graph_revision": self.graph_revision,
            "horizon": self.horizon,
            "lower_quantile": _hex(self.lower_quantile),
            "model_identity": self.model_identity,
            "regime": self.regime,
            "schema": CONFORMAL_KEY_SCHEMA,
            "target_count": self.target_count,
            "upper_quantile": _hex(self.upper_quantile),
        }

    @classmethod
    def from_record(cls, value: object) -> ConformalFieldKey:
        fields = {
            "action",
            "feature_schema",
            "graph_revision",
            "horizon",
            "lower_quantile",
            "model_identity",
            "regime",
            "schema",
            "target_count",
            "upper_quantile",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ConformalIntegrityError("conformal key fields are invalid")
        if value.get("schema") != CONFORMAL_KEY_SCHEMA:
            raise ConformalIntegrityError("conformal key schema is unsupported")
        try:
            result = cls(
                model_identity=value["model_identity"],
                feature_schema=value["feature_schema"],
                graph_revision=value["graph_revision"],
                regime=value["regime"],
                action=value["action"],
                horizon=value["horizon"],
                target_count=value["target_count"],
                lower_quantile=_from_hex(
                    value["lower_quantile"], field="lower_quantile"
                ),
                upper_quantile=_from_hex(
                    value["upper_quantile"], field="upper_quantile"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ConformalIntegrityError("conformal key values are invalid") from exc
        if result.to_record() != dict(value):
            raise ConformalIntegrityError("conformal key is not canonical")
        return result

    @property
    def exact_id(self) -> str:
        return sha256_json(self.to_record())

    @property
    def pooled_record(self) -> dict[str, object]:
        return {
            "feature_schema": self.feature_schema,
            "horizon": self.horizon,
            "lower_quantile": _hex(self.lower_quantile),
            "model_identity": self.model_identity,
            "target_count": self.target_count,
            "upper_quantile": _hex(self.upper_quantile),
        }

    @property
    def pooled_id(self) -> str:
        return sha256_json(self.pooled_record)


@dataclass(frozen=True, slots=True)
class ConformalApplication:
    """Prior-only calibration result and its uncalibrated source forecast."""

    key: ConformalFieldKey
    base_prediction: ForecastPrediction
    prediction: ForecastPrediction
    scale: object
    lower_score_radius: object
    upper_score_radius: object
    lower_expansion: object
    upper_expansion: object
    joint_score_radius: float
    joint_lower: object
    joint_upper: object
    lower_alpha: float
    upper_alpha: float
    joint_alpha: float
    prior_settlements: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, ConformalFieldKey):
            raise TypeError("key must be ConformalFieldKey")
        if not isinstance(self.base_prediction, ForecastPrediction) or not isinstance(
            self.prediction, ForecastPrediction
        ):
            raise TypeError("predictions must be ForecastPrediction instances")
        if (
            self.base_prediction.series_id != self.prediction.series_id
            or self.base_prediction.horizon != self.key.horizon
            or self.base_prediction.target_dim != self.key.target_count
            or not np.array_equal(
                self.base_prediction.timestamps,
                self.prediction.timestamps,
            )
        ):
            raise ValueError("application predictions do not match its key")
        for field in (
            "scale",
            "lower_score_radius",
            "upper_score_radius",
            "lower_expansion",
            "upper_expansion",
            "joint_lower",
            "joint_upper",
        ):
            object.__setattr__(
                self,
                field,
                _readonly_float32(
                    getattr(self, field),
                    shape=self.key.shape,
                    field=field,
                ),
            )
        for field in (
            "joint_score_radius",
            "lower_alpha",
            "upper_alpha",
            "joint_alpha",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field=field, minimum=0.0),
            )
        object.__setattr__(
            self,
            "prior_settlements",
            _nonnegative_int(
                self.prior_settlements,
                field="prior_settlements",
                maximum=2**63 - 1,
            ),
        )


@dataclass(frozen=True, slots=True)
class ConformalUpdate:
    """Committed settlement summary after one atomic field update."""

    observation_id: str | None
    content_sha256: str
    observed_values: int
    marginal_hits: int
    joint_observed: bool
    joint_hit: bool | None
    lower_alpha: float
    upper_alpha: float
    joint_alpha: float
    gain: float
    mean_width_expansion: float
    exact_settlements: int
    replayed: bool = False

    def __post_init__(self) -> None:
        if self.observation_id is not None:
            object.__setattr__(
                self,
                "observation_id",
                _canonical_text(self.observation_id, field="observation_id"),
            )
        if (
            not isinstance(self.content_sha256, str)
            or len(self.content_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.content_sha256
            )
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        for field in ("observed_values", "marginal_hits", "exact_settlements"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.marginal_hits > self.observed_values:
            raise ValueError("marginal_hits cannot exceed observed_values")
        if not isinstance(self.joint_observed, bool) or not isinstance(
            self.replayed, bool
        ):
            raise TypeError("joint_observed and replayed must be boolean")
        if self.joint_hit is not None and not isinstance(self.joint_hit, bool):
            raise TypeError("joint_hit must be boolean or None")
        if self.joint_observed != (self.joint_hit is not None):
            raise ValueError("joint observation and hit fields disagree")
        for field in (
            "lower_alpha",
            "upper_alpha",
            "joint_alpha",
            "gain",
            "mean_width_expansion",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field=field, minimum=0.0),
            )

    def to_record(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "exact_settlements": self.exact_settlements,
            "gain": _hex(self.gain),
            "joint_alpha": _hex(self.joint_alpha),
            "joint_hit": self.joint_hit,
            "joint_observed": self.joint_observed,
            "lower_alpha": _hex(self.lower_alpha),
            "marginal_hits": self.marginal_hits,
            "mean_width_expansion": _hex(self.mean_width_expansion),
            "observation_id": self.observation_id,
            "observed_values": self.observed_values,
            "replayed": False,
            "upper_alpha": _hex(self.upper_alpha),
        }

    @classmethod
    def from_record(cls, value: object, *, replayed: bool) -> ConformalUpdate:
        fields = {
            "content_sha256",
            "exact_settlements",
            "gain",
            "joint_alpha",
            "joint_hit",
            "joint_observed",
            "lower_alpha",
            "marginal_hits",
            "mean_width_expansion",
            "observation_id",
            "observed_values",
            "replayed",
            "upper_alpha",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value["replayed"] is not False
        ):
            raise ConformalIntegrityError("conformal update record is invalid")
        try:
            return cls(
                observation_id=value["observation_id"],
                content_sha256=value["content_sha256"],
                observed_values=value["observed_values"],
                marginal_hits=value["marginal_hits"],
                joint_observed=value["joint_observed"],
                joint_hit=value["joint_hit"],
                lower_alpha=_from_hex(value["lower_alpha"], field="lower_alpha"),
                upper_alpha=_from_hex(value["upper_alpha"], field="upper_alpha"),
                joint_alpha=_from_hex(value["joint_alpha"], field="joint_alpha"),
                gain=_from_hex(value["gain"], field="gain"),
                mean_width_expansion=_from_hex(
                    value["mean_width_expansion"], field="mean_width_expansion"
                ),
                exact_settlements=value["exact_settlements"],
                replayed=replayed,
            )
        except (TypeError, ValueError) as exc:
            raise ConformalIntegrityError(
                "conformal update values are invalid"
            ) from exc


@dataclass(slots=True)
class _Profile:
    lower_alpha: float
    upper_alpha: float
    joint_alpha: float
    lower_history: list[list[float | None]]
    upper_history: list[list[float | None]]
    joint_history: list[float | None]
    settlements: int = 0
    observed_values: int = 0
    marginal_hits: int = 0
    joint_count: int = 0
    joint_hits: int = 0
    width_expansion_sum: float = 0.0

    @classmethod
    def empty(
        cls,
        cells: int,
        *,
        lower_alpha: float,
        upper_alpha: float,
        joint_alpha: float,
    ) -> _Profile:
        return cls(
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
            joint_alpha=joint_alpha,
            lower_history=[[] for _ in range(cells)],
            upper_history=[[] for _ in range(cells)],
            joint_history=[],
        )

    def to_record(self, *, identity: Mapping[str, object]) -> dict[str, object]:
        return {
            "identity": dict(identity),
            "joint_alpha": _hex(self.joint_alpha),
            "joint_count": self.joint_count,
            "joint_history": _history_to_record(self.joint_history),
            "joint_hits": self.joint_hits,
            "lower_alpha": _hex(self.lower_alpha),
            "lower_history": [
                _history_to_record(history) for history in self.lower_history
            ],
            "marginal_hits": self.marginal_hits,
            "observed_values": self.observed_values,
            "settlements": self.settlements,
            "upper_alpha": _hex(self.upper_alpha),
            "upper_history": [
                _history_to_record(history) for history in self.upper_history
            ],
            "width_expansion_sum": _hex(self.width_expansion_sum),
        }

    @classmethod
    def from_record(cls, value: object, *, cells: int, max_history: int) -> _Profile:
        fields = {
            "identity",
            "joint_alpha",
            "joint_count",
            "joint_history",
            "joint_hits",
            "lower_alpha",
            "lower_history",
            "marginal_hits",
            "observed_values",
            "settlements",
            "upper_alpha",
            "upper_history",
            "width_expansion_sum",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ConformalIntegrityError("conformal profile fields are invalid")
        lower_raw = value["lower_history"]
        upper_raw = value["upper_history"]
        if (
            not isinstance(lower_raw, list)
            or not isinstance(upper_raw, list)
            or len(lower_raw) != cells
            or len(upper_raw) != cells
        ):
            raise ConformalIntegrityError("conformal cell history shape is invalid")
        lower = [
            _history_from_record(history, field="lower history")
            for history in lower_raw
        ]
        upper = [
            _history_from_record(history, field="upper history")
            for history in upper_raw
        ]
        joint = _history_from_record(value["joint_history"], field="joint history")
        if (
            any(len(history) > max_history for history in (*lower, *upper))
            or len(joint) > max_history
        ):
            raise ConformalIntegrityError(
                "conformal history exceeds its configured cap"
            )
        integer_fields = (
            "settlements",
            "observed_values",
            "marginal_hits",
            "joint_count",
            "joint_hits",
        )
        for field in integer_fields:
            item = value[field]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ConformalIntegrityError(f"{field} is invalid")
        if (
            value["marginal_hits"] > value["observed_values"]
            or value["joint_hits"] > value["joint_count"]
        ):
            raise ConformalIntegrityError("conformal hit counters are invalid")
        return cls(
            lower_alpha=_from_hex(value["lower_alpha"], field="lower_alpha"),
            upper_alpha=_from_hex(value["upper_alpha"], field="upper_alpha"),
            joint_alpha=_from_hex(value["joint_alpha"], field="joint_alpha"),
            lower_history=lower,
            upper_history=upper,
            joint_history=joint,
            settlements=value["settlements"],
            observed_values=value["observed_values"],
            marginal_hits=value["marginal_hits"],
            joint_count=value["joint_count"],
            joint_hits=value["joint_hits"],
            width_expansion_sum=_from_hex(
                value["width_expansion_sum"],
                field="width_expansion_sum",
                nonnegative=True,
            ),
        )


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_RDWR
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:  # pragma: no cover - defensive OS failure
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_load(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {item}")
            ),
        )
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConformalIntegrityError("conformal field is invalid JSON") from exc
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != raw:
        raise ConformalIntegrityError("conformal field is not canonical JSON")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _empirical_higher(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    count = len(values)
    rank = min(count, max(1, math.ceil((count + 1) * probability)))
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    result = float(ordered[rank - 1])
    return max(result, 0.0)


def _observed(history: Sequence[float | None]) -> list[float]:
    return [value for value in history if value is not None]


def _shrink(local: float, local_n: int, parent: float, weight: float) -> float:
    if local_n == 0:
        return parent
    if weight == 0.0:
        return local
    return (local_n * local + weight * parent) / (local_n + weight)


def _clip_alpha(value: float, config: ConformalFieldConfig) -> float:
    return min(max(value, config.alpha_min), config.alpha_max)


class O1ConformalField:
    """Online conformal field with exact profiles and transferable pooled memory."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        config: ConformalFieldConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.state_path = self.root / _STATE_FILE
        self.lock_path = self.root / _LOCK_FILE
        self.config = config or ConformalFieldConfig()
        if not isinstance(self.config, ConformalFieldConfig):
            raise TypeError("config must be ConformalFieldConfig")
        self._mutex = _lock_for(self.state_path)
        self._exact: dict[str, tuple[ConformalFieldKey, _Profile]] = {}
        self._pooled: dict[str, tuple[dict[str, object], _Profile]] = {}
        self._observations: dict[str, tuple[str, dict[str, object]]] = {}
        with self._mutex:
            if self.state_path.exists():
                self._load()

    @property
    def side_target(self) -> float:
        return (1.0 - self.config.target_coverage) / 2.0

    @property
    def joint_target(self) -> float:
        return 1.0 - self.config.target_coverage

    def _initial_alphas(self) -> tuple[float, float, float]:
        return (
            _clip_alpha(self.side_target, self.config),
            _clip_alpha(self.side_target, self.config),
            _clip_alpha(self.joint_target, self.config),
        )

    def _get_profiles(
        self, key: ConformalFieldKey
    ) -> tuple[_Profile | None, _Profile | None]:
        exact = self._exact.get(key.exact_id)
        pooled = self._pooled.get(key.pooled_id)
        return (
            None if exact is None else exact[1],
            None if pooled is None else pooled[1],
        )

    def _ensure_profiles(self, key: ConformalFieldKey) -> tuple[_Profile, _Profile]:
        cells = key.horizon * key.target_count
        exact, pooled = self._get_profiles(key)
        if pooled is None:
            alphas = self._initial_alphas()
            pooled = _Profile.empty(
                cells,
                lower_alpha=alphas[0],
                upper_alpha=alphas[1],
                joint_alpha=alphas[2],
            )
            self._pooled[key.pooled_id] = (key.pooled_record, pooled)
        if exact is None:
            exact = _Profile.empty(
                cells,
                lower_alpha=pooled.lower_alpha,
                upper_alpha=pooled.upper_alpha,
                joint_alpha=pooled.joint_alpha,
            )
            self._exact[key.exact_id] = (key, exact)
        return exact, pooled

    def _quantile_indexes(
        self, key: ConformalFieldKey, prediction: ForecastPrediction
    ) -> tuple[int, int]:
        if (
            prediction.horizon != key.horizon
            or prediction.target_dim != key.target_count
        ):
            raise ValueError("prediction shape does not match conformal key")
        if prediction.quantiles is None or prediction.quantile_levels is None:
            raise ValueError("conformal calibration requires quantile forecasts")
        levels = np.asarray(prediction.quantile_levels, dtype=np.float64)
        lower_matches = np.flatnonzero(
            np.isclose(levels, key.lower_quantile, atol=1e-7, rtol=0.0)
        )
        upper_matches = np.flatnonzero(
            np.isclose(levels, key.upper_quantile, atol=1e-7, rtol=0.0)
        )
        if lower_matches.size != 1 or upper_matches.size != 1:
            raise ValueError("key outer quantiles are absent from prediction")
        lower_index = int(lower_matches[0])
        upper_index = int(upper_matches[0])
        if lower_index != 0 or upper_index != len(levels) - 1:
            raise ValueError("key quantiles must be the prediction's outer quantiles")
        return lower_index, upper_index

    def _cell_radii(
        self,
        exact: _Profile | None,
        pooled: _Profile | None,
        *,
        lower: bool,
        alpha: float,
        cells: int,
    ) -> NDArray[np.float64]:
        probability = min(max(1.0 - alpha, 0.0), 1.0)
        pooled_histories = (
            [[] for _ in range(cells)]
            if pooled is None
            else (pooled.lower_history if lower else pooled.upper_history)
        )
        exact_histories = (
            [[] for _ in range(cells)]
            if exact is None
            else (exact.lower_history if lower else exact.upper_history)
        )
        all_values = [
            value
            for history in pooled_histories
            for value in history
            if value is not None
        ]
        all_radius = _empirical_higher(all_values, probability)
        result = np.zeros(cells, dtype=np.float64)
        weight = self.config.local_shrinkage
        for index in range(cells):
            pool_values = _observed(pooled_histories[index])
            exact_values = _observed(exact_histories[index])
            pool_radius = _empirical_higher(pool_values, probability)
            pooled_radius = _shrink(
                pool_radius,
                len(pool_values),
                all_radius,
                weight,
            )
            exact_radius = _empirical_higher(exact_values, probability)
            result[index] = _shrink(
                exact_radius,
                len(exact_values),
                pooled_radius,
                weight,
            )
        return result

    def _joint_radius(
        self,
        exact: _Profile | None,
        pooled: _Profile | None,
        *,
        alpha: float,
    ) -> float:
        probability = min(max(1.0 - alpha, 0.0), 1.0)
        pooled_values = [] if pooled is None else _observed(pooled.joint_history)
        exact_values = [] if exact is None else _observed(exact.joint_history)
        pooled_radius = _empirical_higher(pooled_values, probability)
        exact_radius = _empirical_higher(exact_values, probability)
        return _shrink(
            exact_radius,
            len(exact_values),
            pooled_radius,
            self.config.local_shrinkage,
        )

    def apply(
        self,
        key: ConformalFieldKey,
        base: ForecastPrediction,
    ) -> ConformalApplication:
        """Expand only the base outer quantiles using committed prior evidence."""

        if not isinstance(key, ConformalFieldKey):
            raise TypeError("key must be ConformalFieldKey")
        if not isinstance(base, ForecastPrediction):
            raise TypeError("base must be ForecastPrediction")
        lower_index, upper_index = self._quantile_indexes(key, base)
        with self._mutex, _file_lock(self.lock_path):
            if self.state_path.exists():
                self._load()
            exact, pooled = self._get_profiles(key)
            if exact is None:
                if pooled is None:
                    lower_alpha, upper_alpha, joint_alpha = self._initial_alphas()
                else:
                    lower_alpha = pooled.lower_alpha
                    upper_alpha = pooled.upper_alpha
                    joint_alpha = pooled.joint_alpha
                prior_settlements = 0
            else:
                lower_alpha = exact.lower_alpha
                upper_alpha = exact.upper_alpha
                joint_alpha = exact.joint_alpha
                prior_settlements = exact.settlements
            cells = key.horizon * key.target_count
            lower_radius = self._cell_radii(
                exact,
                pooled,
                lower=True,
                alpha=lower_alpha,
                cells=cells,
            ).reshape(key.shape)
            upper_radius = self._cell_radii(
                exact,
                pooled,
                lower=False,
                alpha=upper_alpha,
                cells=cells,
            ).reshape(key.shape)
            joint_radius = self._joint_radius(
                exact,
                pooled,
                alpha=joint_alpha,
            )

        base_quantiles = np.asarray(base.quantiles, dtype=np.float64)
        base_lower = base_quantiles[:, :, lower_index]
        base_upper = base_quantiles[:, :, upper_index]
        scale = np.maximum(
            (base_upper - base_lower) / 2.0,
            self.config.min_scale
            * (1.0 + np.abs(np.asarray(base.point, dtype=np.float64))),
        )
        lower_expansion = np.maximum(lower_radius, 0.0) * scale
        upper_expansion = np.maximum(upper_radius, 0.0) * scale
        quantiles = np.array(base_quantiles, dtype=np.float64, copy=True)
        quantiles[:, :, lower_index] = base_lower - lower_expansion
        quantiles[:, :, upper_index] = base_upper + upper_expansion
        calibrated = ForecastPrediction(
            series_id=base.series_id,
            timestamps=base.timestamps,
            point=base.point,
            quantiles=quantiles,
            quantile_levels=base.quantile_levels,
        )
        joint_expansion = max(joint_radius, 0.0) * scale
        return ConformalApplication(
            key=key,
            base_prediction=base,
            prediction=calibrated,
            scale=scale,
            lower_score_radius=lower_radius,
            upper_score_radius=upper_radius,
            lower_expansion=lower_expansion,
            upper_expansion=upper_expansion,
            joint_score_radius=joint_radius,
            joint_lower=base_lower - joint_expansion,
            joint_upper=base_upper + joint_expansion,
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
            joint_alpha=joint_alpha,
            prior_settlements=prior_settlements,
        )

    def _content_hash(
        self,
        key: ConformalFieldKey,
        application: ConformalApplication,
        target: ForecastTarget,
        *,
        surprise: float,
        cusum: float,
    ) -> str:
        base = application.base_prediction
        return sha256_json(
            {
                "application": {
                    "joint_score_radius": _hex(application.joint_score_radius),
                    "lower_expansion": np.asarray(application.lower_expansion).tolist(),
                    "lower_score_radius": np.asarray(
                        application.lower_score_radius
                    ).tolist(),
                    "scale": np.asarray(application.scale).tolist(),
                    "upper_expansion": np.asarray(application.upper_expansion).tolist(),
                    "upper_score_radius": np.asarray(
                        application.upper_score_radius
                    ).tolist(),
                },
                "base": {
                    "point": np.asarray(base.point).tolist(),
                    "quantile_levels": np.asarray(base.quantile_levels).tolist(),
                    "quantiles": np.asarray(base.quantiles).tolist(),
                    "series_id": base.series_id,
                    "timestamps_ns": [
                        int(item)
                        for item in base.timestamps.astype("datetime64[ns]").astype(
                            np.int64
                        )
                    ],
                },
                "cusum": _hex(cusum),
                "key": key.to_record(),
                "surprise": _hex(surprise),
                "target": {
                    "mask": np.asarray(target.mask).tolist(),
                    "series_id": target.series_id,
                    "timestamps_ns": [
                        int(item)
                        for item in target.timestamps.astype("datetime64[ns]").astype(
                            np.int64
                        )
                    ],
                    "values": np.asarray(target.values).tolist(),
                },
            }
        )

    def _gain(self, *, surprise: float, cusum: float) -> float:
        surprise_factor = 1.0 + self.config.surprise_gain * min(
            surprise,
            self.config.surprise_cap,
        )
        cusum_factor = 1.0 + 0.10 * math.tanh(cusum)
        return self.config.base_gain * surprise_factor * cusum_factor

    def _append_profile(
        self,
        profile: _Profile,
        *,
        lower_scores: NDArray[np.float64],
        upper_scores: NDArray[np.float64],
        mask: NDArray[np.bool_],
        joint_score: float | None,
        lower_misses: int,
        upper_misses: int,
        marginal_hits: int,
        joint_hit: bool | None,
        width_expansion: float,
        gain: float,
    ) -> None:
        flat_mask = mask.reshape(-1)
        for index, observed in enumerate(flat_mask):
            profile.lower_history[index].append(
                float(lower_scores.reshape(-1)[index]) if observed else None
            )
            profile.upper_history[index].append(
                float(upper_scores.reshape(-1)[index]) if observed else None
            )
            if len(profile.lower_history[index]) > self.config.max_history:
                del profile.lower_history[index][: -self.config.max_history]
                del profile.upper_history[index][: -self.config.max_history]
        profile.joint_history.append(joint_score)
        if len(profile.joint_history) > self.config.max_history:
            del profile.joint_history[: -self.config.max_history]
        observed_count = int(mask.sum())
        if observed_count:
            profile.lower_alpha = _clip_alpha(
                profile.lower_alpha
                + gain * (self.side_target - lower_misses / observed_count),
                self.config,
            )
            profile.upper_alpha = _clip_alpha(
                profile.upper_alpha
                + gain * (self.side_target - upper_misses / observed_count),
                self.config,
            )
            joint_miss = float(not bool(joint_hit))
            profile.joint_alpha = _clip_alpha(
                profile.joint_alpha + gain * (self.joint_target - joint_miss),
                self.config,
            )
        profile.settlements += 1
        profile.observed_values += observed_count
        profile.marginal_hits += marginal_hits
        if joint_score is not None:
            profile.joint_count += 1
            profile.joint_hits += int(bool(joint_hit))
        profile.width_expansion_sum += width_expansion

    def observe(
        self,
        key: ConformalFieldKey,
        application: ConformalApplication,
        target: ForecastTarget,
        *,
        surprise: float = 0.0,
        cusum: float = 0.0,
        observation_id: str | None = None,
    ) -> ConformalUpdate:
        """Settle one forecast and atomically update exact and pooled fields."""

        if not isinstance(key, ConformalFieldKey):
            raise TypeError("key must be ConformalFieldKey")
        if not isinstance(application, ConformalApplication):
            raise TypeError("application must be ConformalApplication")
        if not isinstance(target, ForecastTarget):
            raise TypeError("target must be ForecastTarget")
        if application.key != key:
            raise ValueError("application belongs to a different conformal key")
        base = application.base_prediction
        if (
            target.series_id != base.series_id
            or target.horizon != key.horizon
            or target.target_dim != key.target_count
            or not np.array_equal(target.timestamps, base.timestamps)
        ):
            raise ValueError("target does not align with the conformal application")
        surprise_value = _finite(surprise, field="surprise", minimum=0.0)
        cusum_value = _finite(cusum, field="cusum", minimum=0.0)
        if observation_id is not None:
            observation_id = _canonical_text(observation_id, field="observation_id")
        content_sha256 = self._content_hash(
            key,
            application,
            target,
            surprise=surprise_value,
            cusum=cusum_value,
        )
        lower_index, upper_index = self._quantile_indexes(key, base)
        base_quantiles = np.asarray(base.quantiles, dtype=np.float64)
        lower = base_quantiles[:, :, lower_index]
        upper = base_quantiles[:, :, upper_index]
        actual = np.asarray(target.values, dtype=np.float64)
        mask = np.asarray(target.mask, dtype=np.bool_)
        scale = np.asarray(application.scale, dtype=np.float64)
        lower_scores = np.maximum((lower - actual) / scale, 0.0)
        upper_scores = np.maximum((actual - upper) / scale, 0.0)
        lower_scores[~mask] = 0.0
        upper_scores[~mask] = 0.0
        observed_count = int(mask.sum())
        calibrated_quantiles = np.asarray(
            application.prediction.quantiles,
            dtype=np.float64,
        )
        calibrated_lower = calibrated_quantiles[:, :, lower_index]
        calibrated_upper = calibrated_quantiles[:, :, upper_index]
        lower_miss_mask = (actual < calibrated_lower) & mask
        upper_miss_mask = (actual > calibrated_upper) & mask
        marginal_hit_mask = (
            (actual >= calibrated_lower) & (actual <= calibrated_upper) & mask
        )
        marginal_hits = int(marginal_hit_mask.sum())
        joint_score = (
            None
            if observed_count == 0
            else float(max(lower_scores[mask].max(), upper_scores[mask].max()))
        )
        joint_hit = (
            None
            if observed_count == 0
            else bool(
                (
                    (actual[mask] >= np.asarray(application.joint_lower)[mask])
                    & (actual[mask] <= np.asarray(application.joint_upper)[mask])
                ).all()
            )
        )
        width_expansion = float(
            (
                np.asarray(application.lower_expansion, dtype=np.float64)
                + np.asarray(application.upper_expansion, dtype=np.float64)
            )[mask].sum()
        )
        gain = self._gain(surprise=surprise_value, cusum=cusum_value)

        with self._mutex, _file_lock(self.lock_path):
            if self.state_path.exists():
                self._load()
            if observation_id is not None and observation_id in self._observations:
                previous_hash, previous_update = self._observations[observation_id]
                if previous_hash != content_sha256:
                    raise ConformalConflictError(
                        "observation ID was replayed with different evidence"
                    )
                return ConformalUpdate.from_record(previous_update, replayed=True)
            if (
                observation_id is not None
                and len(self._observations) >= _MAX_OBSERVATION_IDS
            ):
                raise ConformalFieldError("conformal observation index is full")
            exact, pooled = self._ensure_profiles(key)
            self._append_profile(
                exact,
                lower_scores=lower_scores,
                upper_scores=upper_scores,
                mask=mask,
                joint_score=joint_score,
                lower_misses=int(lower_miss_mask.sum()),
                upper_misses=int(upper_miss_mask.sum()),
                marginal_hits=marginal_hits,
                joint_hit=joint_hit,
                width_expansion=width_expansion,
                gain=gain,
            )
            self._append_profile(
                pooled,
                lower_scores=lower_scores,
                upper_scores=upper_scores,
                mask=mask,
                joint_score=joint_score,
                lower_misses=int(lower_miss_mask.sum()),
                upper_misses=int(upper_miss_mask.sum()),
                marginal_hits=marginal_hits,
                joint_hit=joint_hit,
                width_expansion=width_expansion,
                gain=gain,
            )
            update = ConformalUpdate(
                observation_id=observation_id,
                content_sha256=content_sha256,
                observed_values=observed_count,
                marginal_hits=marginal_hits,
                joint_observed=joint_score is not None,
                joint_hit=joint_hit,
                lower_alpha=exact.lower_alpha,
                upper_alpha=exact.upper_alpha,
                joint_alpha=exact.joint_alpha,
                gain=gain,
                mean_width_expansion=(
                    width_expansion / observed_count if observed_count else 0.0
                ),
                exact_settlements=exact.settlements,
            )
            if observation_id is not None:
                self._observations[observation_id] = (
                    content_sha256,
                    update.to_record(),
                )
            self._persist()
            return update

    def metrics(self) -> dict[str, int | float]:
        """Return aggregate settlement and coverage metrics without double counting pools."""

        with self._mutex, _file_lock(self.lock_path):
            if self.state_path.exists():
                self._load()
            profiles = [profile for _key, profile in self._exact.values()]
            settlements = sum(profile.settlements for profile in profiles)
            observed_values = sum(profile.observed_values for profile in profiles)
            marginal_hits = sum(profile.marginal_hits for profile in profiles)
            joint_count = sum(profile.joint_count for profile in profiles)
            joint_hits = sum(profile.joint_hits for profile in profiles)
            width_sum = sum(profile.width_expansion_sum for profile in profiles)
            return {
                "exact_profile_count": len(self._exact),
                "pooled_profile_count": len(self._pooled),
                "settlements": settlements,
                "observed_values": observed_values,
                "mean_marginal_coverage": (
                    marginal_hits / observed_values if observed_values else 0.0
                ),
                "joint_coverage": joint_hits / joint_count if joint_count else 0.0,
                "mean_marginal_width_expansion": (
                    width_sum / observed_values if observed_values else 0.0
                ),
            }

    def _document(self) -> dict[str, object]:
        return {
            "config": self.config.to_record(),
            "exact": {
                identity: profile.to_record(identity=key.to_record())
                for identity, (key, profile) in sorted(self._exact.items())
            },
            "observations": {
                observation_id: {
                    "content_sha256": content_sha256,
                    "update": update,
                }
                for observation_id, (content_sha256, update) in sorted(
                    self._observations.items()
                )
            },
            "pooled": {
                identity: profile.to_record(identity=record)
                for identity, (record, profile) in sorted(self._pooled.items())
            },
            "schema": CONFORMAL_FIELD_SCHEMA,
        }

    def _persist(self) -> None:
        _atomic_write(self.state_path, canonical_json_bytes(self._document()))

    def _load(self) -> None:
        document = _strict_load(self.state_path)
        if set(document) != {"config", "exact", "observations", "pooled", "schema"}:
            raise ConformalIntegrityError("conformal field fields are invalid")
        if document.get("schema") != CONFORMAL_FIELD_SCHEMA:
            raise ConformalIntegrityError("conformal field schema is unsupported")
        stored_config = ConformalFieldConfig.from_record(document["config"])
        if stored_config != self.config:
            raise ConformalIntegrityError(
                "conformal field config differs from requested config"
            )
        exact_raw = document["exact"]
        pooled_raw = document["pooled"]
        observations_raw = document["observations"]
        if (
            not isinstance(exact_raw, Mapping)
            or not isinstance(pooled_raw, Mapping)
            or not isinstance(observations_raw, Mapping)
        ):
            raise ConformalIntegrityError("conformal field indexes are invalid")
        exact: dict[str, tuple[ConformalFieldKey, _Profile]] = {}
        for identity, raw in exact_raw.items():
            if not isinstance(identity, str) or not isinstance(raw, Mapping):
                raise ConformalIntegrityError("exact conformal profile is invalid")
            key = ConformalFieldKey.from_record(raw.get("identity"))
            if identity != key.exact_id:
                raise ConformalIntegrityError(
                    "exact conformal profile identity differs"
                )
            exact[identity] = (
                key,
                _Profile.from_record(
                    raw,
                    cells=key.horizon * key.target_count,
                    max_history=self.config.max_history,
                ),
            )
        pooled: dict[str, tuple[dict[str, object], _Profile]] = {}
        for identity, raw in pooled_raw.items():
            if not isinstance(identity, str) or not isinstance(raw, Mapping):
                raise ConformalIntegrityError("pooled conformal profile is invalid")
            record = raw.get("identity")
            if not isinstance(record, Mapping):
                raise ConformalIntegrityError("pooled conformal identity is invalid")
            required = {
                "feature_schema",
                "horizon",
                "lower_quantile",
                "model_identity",
                "target_count",
                "upper_quantile",
            }
            if set(record) != required or identity != sha256_json(record):
                raise ConformalIntegrityError(
                    "pooled conformal profile identity differs"
                )
            try:
                horizon = _positive_int(
                    record["horizon"], field="horizon", maximum=_MAX_AXIS
                )
                target_count = _positive_int(
                    record["target_count"], field="target_count", maximum=_MAX_AXIS
                )
            except ValueError as exc:
                raise ConformalIntegrityError(
                    "pooled conformal shape is invalid"
                ) from exc
            pooled[identity] = (
                dict(record),
                _Profile.from_record(
                    raw,
                    cells=horizon * target_count,
                    max_history=self.config.max_history,
                ),
            )
        observations: dict[str, tuple[str, dict[str, object]]] = {}
        if len(observations_raw) > _MAX_OBSERVATION_IDS:
            raise ConformalIntegrityError("conformal observation index exceeds its cap")
        for observation_id, raw in observations_raw.items():
            try:
                canonical_id = _canonical_text(observation_id, field="observation_id")
            except ValueError as exc:
                raise ConformalIntegrityError("observation ID is invalid") from exc
            if not isinstance(raw, Mapping) or set(raw) != {"content_sha256", "update"}:
                raise ConformalIntegrityError("conformal observation record is invalid")
            update = ConformalUpdate.from_record(raw["update"], replayed=False)
            if (
                update.observation_id != canonical_id
                or update.content_sha256 != raw["content_sha256"]
            ):
                raise ConformalIntegrityError("conformal observation record differs")
            observations[canonical_id] = (update.content_sha256, update.to_record())
        self._exact = exact
        self._pooled = pooled
        self._observations = observations


__all__ = [
    "CONFORMAL_FIELD_SCHEMA",
    "ConformalApplication",
    "ConformalConflictError",
    "ConformalFieldConfig",
    "ConformalFieldError",
    "ConformalFieldKey",
    "ConformalIntegrityError",
    "ConformalUpdate",
    "O1ConformalField",
]
