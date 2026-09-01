"""Canonical identities shared by forecasting runtime state.

The module intentionally implements a small JSON subset rather than relying on
``default=str`` or NumPy-aware encoders.  A value either has one bounded,
portable byte representation or it is rejected before it can become an
identity, state seal, or revision hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

IDENTITY_SCHEMA = "o1fm.canonical-identity/v1"
MAX_CANONICAL_BYTES = 16 * 1024 * 1024
MAX_CANONICAL_DEPTH = 64
MAX_CANONICAL_ITEMS = 1_000_000
MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_INTEGER = (1 << 63) - 1

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}\Z")


class IdentityError(ValueError):
    """A value cannot participate in a canonical runtime identity."""


def require_sha256(value: object, *, field: str = "sha256") -> str:
    """Return one lowercase SHA-256 digest or reject it."""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IdentityError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_text(value: str, *, field: str) -> str:
    if (
        "\x00" in value
        or unicodedata.normalize("NFC", value) != value
        or len(value.encode("utf-8")) > MAX_TEXT_BYTES
    ):
        raise IdentityError(f"{field} must be bounded NFC text without NUL")
    return value


def _normalize(value: object) -> object:
    remaining = MAX_CANONICAL_ITEMS

    def visit(item: object, *, depth: int, path: str) -> object:
        nonlocal remaining
        if depth > MAX_CANONICAL_DEPTH:
            raise IdentityError("canonical JSON exceeds its nesting bound")
        remaining -= 1
        if remaining < 0:
            raise IdentityError("canonical JSON exceeds its item bound")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            if not -MAX_INTEGER <= item <= MAX_INTEGER:
                raise IdentityError(f"{path} integer exceeds the signed-64 bound")
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise IdentityError(f"{path} must be finite")
            return 0.0 if item == 0.0 else item
        if isinstance(item, str):
            return _canonical_text(item, field=path)
        if isinstance(item, Mapping):
            result: dict[str, object] = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise IdentityError(f"{path} contains a non-string key")
                canonical_key = _canonical_text(key, field=f"{path} key")
                if canonical_key in result:
                    raise IdentityError(f"{path} contains a duplicate key")
                result[canonical_key] = visit(
                    child,
                    depth=depth + 1,
                    path=f"{path}.{canonical_key}",
                )
            return result
        if isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes, bytearray, memoryview),
        ):
            return [
                visit(child, depth=depth + 1, path=f"{path}[{index}]")
                for index, child in enumerate(item)
            ]
        raise IdentityError(f"{path} contains unsupported value {type(item).__name__}")

    return visit(value, depth=0, path="value")


def canonical_json_bytes(value: object) -> bytes:
    """Encode ``value`` into O1FM's single bounded JSON representation."""

    normalized = _normalize(value)
    try:
        encoded = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # Defensive after normalization.
        raise IdentityError("value is not canonical JSON") from exc
    if len(encoded) > MAX_CANONICAL_BYTES:
        raise IdentityError("canonical JSON exceeds its byte bound")
    return encoded


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise TypeError("value must be immutable bytes")
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    """Return the SHA-256 identity of canonical JSON data."""

    return sha256_bytes(canonical_json_bytes(value))


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


@dataclass(frozen=True, slots=True)
class CanonicalIdentity:
    """A namespaced, immutable collection of execution-contract components."""

    namespace: str
    components: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or _NAMESPACE.fullmatch(self.namespace) is None
        ):
            raise IdentityError("identity namespace is not canonical")
        normalized = _normalize(self.components)
        if not isinstance(normalized, dict) or not normalized:
            raise IdentityError("identity components must be a non-empty mapping")
        object.__setattr__(self, "components", _freeze(normalized))

    def to_record(self) -> dict[str, object]:
        return {
            "components": json.loads(canonical_json_bytes(self.components)),
            "namespace": self.namespace,
            "schema": IDENTITY_SCHEMA,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> CanonicalIdentity:
        if not isinstance(value, Mapping) or set(value) != {
            "components",
            "namespace",
            "schema",
        }:
            raise IdentityError("identity record fields are invalid")
        if value.get("schema") != IDENTITY_SCHEMA:
            raise IdentityError("identity schema is unsupported")
        try:
            result = cls(
                namespace=value["namespace"],
                components=value["components"],
            )
        except (TypeError, ValueError) as exc:
            raise IdentityError("identity record is invalid") from exc
        if result.to_record() != dict(value):
            raise IdentityError("identity record is not canonical")
        return result


__all__ = [
    "IDENTITY_SCHEMA",
    "CanonicalIdentity",
    "IdentityError",
    "canonical_json_bytes",
    "require_sha256",
    "sha256_bytes",
    "sha256_json",
]
