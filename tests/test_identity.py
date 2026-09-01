from __future__ import annotations

import math

import pytest

from o1fm.identity import (
    CanonicalIdentity,
    IdentityError,
    canonical_json_bytes,
    require_sha256,
    sha256_json,
)


def test_canonical_json_is_order_stable_and_normalizes_sequences_and_zero() -> None:
    left = {"z": -0.0, "a": (1, 2, {"x": True})}
    right = {"a": [1, 2, {"x": True}], "z": 0.0}

    assert canonical_json_bytes(left) == b'{"a":[1,2,{"x":true}],"z":0.0}'
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == sha256_json(right)


@pytest.mark.parametrize(
    "value",
    [
        {"x": math.nan},
        {"x": math.inf},
        {1: "non-string-key"},
        {"x": b"bytes"},
        {"x": 1 << 80},
        {"x": "not-NFC-e\u0301"},
    ],
)
def test_canonical_json_rejects_nonportable_values(value: object) -> None:
    with pytest.raises(IdentityError):
        canonical_json_bytes(value)


def test_canonical_identity_roundtrips_and_is_immutable() -> None:
    identity = CanonicalIdentity(
        "forecast.model/v1",
        {"horizon": 12, "model": "ridge", "quantiles": [0.1, 0.5, 0.9]},
    )
    restored = CanonicalIdentity.from_record(identity.to_record())

    assert restored == identity
    assert restored.sha256 == identity.sha256
    assert require_sha256(identity.sha256) == identity.sha256
    with pytest.raises(TypeError):
        identity.components["model"] = "mutated"  # type: ignore[index]


def test_identity_rejects_bad_namespace_and_digest() -> None:
    with pytest.raises(IdentityError):
        CanonicalIdentity("bad namespace", {"x": 1})
    with pytest.raises(IdentityError):
        require_sha256("A" * 64)
