from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from o1fm.causal_graph import CausalEdge
from o1fm.transport import (
    causal_transport_stack,
    causal_transport_w1,
    route_transport_w1,
)


def _random_causal_kernel(rng: np.random.Generator, size: int) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float64)
    for row in range(size):
        weights = rng.random(row + 1)
        kernel[row, : row + 1] = weights / weights.sum()
    return kernel


def _edge(
    *, target: str, lag: int, score: float, kind: str = "observational"
) -> CausalEdge:
    return CausalEdge(
        source=f"source-{target}-{lag}",
        target=target,
        lag=lag,
        kind=kind,  # type: ignore[arg-type]
        observations=12,
        covariance=score,
        source_variance=1.0,
        target_variance=1.0,
        score=score,
        revision_sha256="a" * 64,
    )


def test_irregular_sites_match_expected_backward_distance() -> None:
    query = np.array([0.1, 0.2, 0.3, 0.4])
    sites = np.array([-3.0, -1.25, 2.0, 11.0])
    kernel = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.25, 0.75, 0.0, 0.0],
            [0.1, 0.2, 0.7, 0.0],
            [0.15, 0.05, 0.3, 0.5],
        ]
    )

    result = causal_transport_w1(query, kernel, sites)

    assert result.marginal_w1 == pytest.approx(result.expected_backward_distance)
    assert result.equality_error <= 1e-12
    assert result.input_marginal.dtype == np.float64
    assert result.output_marginal.dtype == np.float64
    assert not result.input_marginal.flags.writeable
    assert not result.output_marginal.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.output_marginal[0] = 1.0


@pytest.mark.parametrize("size", [2, 3, 8, 32])
def test_random_causal_kernels_satisfy_exact_identity(size: int) -> None:
    rng = np.random.default_rng(9100 + size)
    for _ in range(20):
        query = rng.random(size)
        query /= query.sum()
        sites = np.cumsum(rng.uniform(0.01, 5.0, size=size))
        result = causal_transport_w1(
            query,
            _random_causal_kernel(rng, size),
            sites,
        )
        assert result.equality_error <= 1e-12 + 1e-12 * result.marginal_w1


def test_causal_stack_is_an_exact_w1_geodesic() -> None:
    rng = np.random.default_rng(2026)
    size = 12
    query = rng.random(size)
    query /= query.sum()
    sites = np.cumsum(rng.lognormal(mean=0.0, sigma=0.6, size=size))
    kernels = tuple(_random_causal_kernel(rng, size) for _ in range(7))

    result = causal_transport_stack(query, kernels, sites)

    assert len(result.layers) == len(kernels)
    assert result.per_layer is result.layers
    assert result.layer_results is result.layers
    assert result.endpoint_w1 == pytest.approx(result.additive_w1, abs=1e-12)
    assert result.equality_error <= 1e-12 + 1e-12 * result.endpoint_w1

    empty = causal_transport_stack(query, (), sites)
    assert empty.layers == ()
    assert empty.endpoint_w1 == 0.0
    assert empty.additive_w1 == 0.0


def test_zero_transport_is_identity_for_full_support_query() -> None:
    query = np.array([0.1, 0.2, 0.3, 0.4])
    identity = causal_transport_w1(query, np.eye(4))
    assert identity.marginal_w1 == 0.0

    rng = np.random.default_rng(4)
    for _ in range(20):
        nonidentity = _random_causal_kernel(rng, query.size)
        assert not np.array_equal(nonidentity, np.eye(query.size))
        moved = causal_transport_w1(query, nonidentity)
        assert moved.marginal_w1 > 0.0


@pytest.mark.parametrize(
    ("query", "kernel", "message"),
    [
        ([0.5, 0.4], np.eye(2), "sum to one"),
        ([0.5, -0.1, 0.6], np.eye(3), "nonnegative"),
        ([0.5, 0.5], [[1.0, 0.0], [-0.1, 1.1]], "nonnegative"),
        ([0.5, 0.5], [[1.0, 0.0], [0.2, 0.7]], "sum to one"),
        ([0.5, 0.5], [[0.9, 0.1], [0.0, 1.0]], "future sites"),
    ],
)
def test_rejects_invalid_probabilities_and_kernels(
    query: list[float], kernel: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        causal_transport_w1(query, kernel)


def test_rejects_nonfinite_and_unordered_sites() -> None:
    with pytest.raises(ValueError, match="finite"):
        causal_transport_w1([0.5, 0.5], np.eye(2), [0.0, np.inf])
    with pytest.raises(ValueError, match="strictly increasing"):
        causal_transport_w1([0.5, 0.5], np.eye(2), [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        causal_transport_w1([np.nan, 0.0], np.eye(2))


def test_route_transport_normalizes_per_target_and_averages_uniformly() -> None:
    edges = (
        _edge(target="demand", lag=2, score=0.25),
        _edge(target="demand", lag=6, score=0.75),
        _edge(target="demand", lag=0, score=1.0, kind="intervention"),
        _edge(target="price", lag=3, score=0.5),
        _edge(target="unused", lag=99, score=1.0),
    )

    result = route_transport_w1(edges, ("demand", "price", "missing"))

    assert isinstance(result.per_target_w1, MappingProxyType)
    assert result.per_target_w1 == {
        "demand": pytest.approx(2.5),
        "price": pytest.approx(3.0),
        "missing": 0.0,
    }
    assert result.mean_w1 == pytest.approx(5.5 / 3.0)
    per_target, mean = result
    assert per_target is result.per_target_w1
    assert mean == result.mean_w1
    with pytest.raises(TypeError):
        result.per_target_w1["new"] = 1.0  # type: ignore[index]


def test_route_transport_handles_empty_and_zero_score_routes() -> None:
    zero = _edge(target="zero", lag=8, score=0.0)
    result = route_transport_w1((zero,), ("zero", "empty"))
    assert result.per_target_w1 == {"zero": 0.0, "empty": 0.0}
    assert result.mean_w1 == 0.0

    no_targets = route_transport_w1((), ())
    assert dict(no_targets.per_target_w1) == {}
    assert no_targets.mean_w1 == 0.0
