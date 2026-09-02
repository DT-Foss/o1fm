"""Exact causal transport geometry on ordered supports.

For a probability row vector ``q`` and a row-stochastic lower-triangular
kernel ``A``, the induced marginal ``u = q @ A`` satisfies

``W1(q, u) = sum(q[i] * A[i, j] * (x[i] - x[j]))``.

The implementation evaluates both sides independently and refuses to return
if floating-point arithmetic violates the identity beyond the public numerical
tolerance below.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .causal_graph import CausalEdge

TRANSPORT_ATOL = 1e-12
TRANSPORT_RTOL = 1e-12


def _tolerance(left: float, right: float) -> float:
    return TRANSPORT_ATOL + TRANSPORT_RTOL * max(abs(left), abs(right))


def _immutable_float64(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _probability(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = _immutable_float64(values, name=name)
    if bool(np.any(result < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    total = math.fsum(float(value) for value in result)
    if abs(total - 1.0) > _tolerance(total, 1.0):
        raise ValueError(f"{name} must sum to one")
    return result


def _ordered_sites(sites: ArrayLike | None, *, size: int) -> NDArray[np.float64]:
    selected: ArrayLike = np.arange(size, dtype=np.float64) if sites is None else sites
    result = _immutable_float64(selected, name="sites")
    if result.shape != (size,):
        raise ValueError("sites must match the query length")
    if not bool(np.all(np.diff(result) > 0.0)):
        raise ValueError("sites must be strictly increasing")
    return result


def _causal_kernel(kernel: ArrayLike, *, size: int) -> NDArray[np.float64]:
    result = np.array(kernel, dtype=np.float64, order="C", copy=True)
    if result.ndim != 2 or result.shape != (size, size):
        raise ValueError("kernel must be square and match the query length")
    if not bool(np.isfinite(result).all()):
        raise ValueError("kernel must be finite")
    if bool(np.any(result < 0.0)):
        raise ValueError("kernel must be nonnegative")
    if bool(np.any(np.triu(result, k=1) != 0.0)):
        raise ValueError("kernel must not transport mass to future sites")
    row_sums = result.sum(axis=1, dtype=np.float64)
    if not bool(
        np.allclose(
            row_sums,
            np.ones(size, dtype=np.float64),
            rtol=TRANSPORT_RTOL,
            atol=TRANSPORT_ATOL,
        )
    ):
        raise ValueError("kernel rows must sum to one")
    result.setflags(write=False)
    return result


def _discrete_w1(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
    sites: NDArray[np.float64],
) -> float:
    """Compute one-dimensional discrete W1 from CDF mass between sites."""
    if left.size == 1:
        return 0.0
    cdf_delta = np.cumsum(left - right, dtype=np.float64)[:-1]
    widths = np.diff(sites)
    return math.fsum(
        abs(float(delta)) * float(width)
        for delta, width in zip(cdf_delta, widths, strict=True)
    )


@dataclass(frozen=True, slots=True)
class CausalTransportResult:
    """One exact causal transport step and its two independent costs."""

    input_marginal: NDArray[np.float64]
    output_marginal: NDArray[np.float64]
    expected_backward_distance: float
    marginal_w1: float
    equality_error: float

    def __post_init__(self) -> None:
        input_marginal = _probability(self.input_marginal, name="input_marginal")
        output_marginal = _probability(self.output_marginal, name="output_marginal")
        if input_marginal.shape != output_marginal.shape:
            raise ValueError("input and output marginals must have equal shape")
        object.__setattr__(self, "input_marginal", input_marginal)
        object.__setattr__(self, "output_marginal", output_marginal)
        for name in (
            "expected_backward_distance",
            "marginal_w1",
            "equality_error",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CausalTransportStackResult:
    """A causal rollout and the independently checked geodesic identity."""

    layers: tuple[CausalTransportResult, ...]
    endpoint_w1: float
    additive_w1: float
    equality_error: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        for name in ("endpoint_w1", "additive_w1", "equality_error"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)

    @property
    def per_layer(self) -> tuple[CausalTransportResult, ...]:
        """Alias emphasizing that ``layers`` contains one result per kernel."""
        return self.layers

    @property
    def layer_results(self) -> tuple[CausalTransportResult, ...]:
        """Descriptive alias for callers that persist result records."""
        return self.layers


@dataclass(frozen=True, slots=True)
class RouteTransportResult:
    """Normalized route-score transport, separated from raw correlations.

    ``per_target_w1`` applies the exact backward-distance identity to the
    probability transport obtained by normalizing each target's positive edge
    scores. It does not assert that raw correlation scores are transport mass.
    """

    per_target_w1: Mapping[str, float]
    mean_w1: float

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for target, value in self.per_target_w1.items():
            if not isinstance(target, str) or not target:
                raise ValueError("route targets must be non-empty strings")
            selected = float(value)
            if not math.isfinite(selected) or selected < 0.0:
                raise ValueError("route W1 values must be finite and nonnegative")
            normalized[target] = selected
        mean = float(self.mean_w1)
        if not math.isfinite(mean) or mean < 0.0:
            raise ValueError("mean_w1 must be finite and nonnegative")
        object.__setattr__(self, "per_target_w1", MappingProxyType(normalized))
        object.__setattr__(self, "mean_w1", mean)

    def __iter__(self) -> Iterator[object]:
        """Support ``per_target, mean = result`` without losing named fields."""
        yield self.per_target_w1
        yield self.mean_w1


def causal_transport_w1(
    query: ArrayLike,
    kernel: ArrayLike,
    sites: ArrayLike | None = None,
) -> CausalTransportResult:
    """Evaluate the exact Foss causal-transport identity for one kernel."""
    input_marginal = _probability(query, name="query")
    selected_sites = _ordered_sites(sites, size=input_marginal.size)
    selected_kernel = _causal_kernel(kernel, size=input_marginal.size)
    # Some Accelerate-linked NumPy builds emit spurious floating warnings for
    # finite BLAS operations. Validation below still rejects a genuinely
    # non-finite result.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        induced = input_marginal @ selected_kernel
        distance = selected_sites[:, None] - selected_sites[None, :]
        coupling_costs = input_marginal[:, None] * selected_kernel * distance
    output_marginal = _probability(induced, name="induced marginal")
    expected = math.fsum(float(value) for value in coupling_costs.flat)
    marginal_w1 = _discrete_w1(
        input_marginal,
        output_marginal,
        selected_sites,
    )
    error = abs(expected - marginal_w1)
    if error > _tolerance(expected, marginal_w1):
        raise ArithmeticError(
            "causal coupling cost and marginal W1 violate the Foss identity"
        )

    return CausalTransportResult(
        input_marginal=input_marginal,
        output_marginal=output_marginal,
        expected_backward_distance=max(0.0, expected),
        marginal_w1=max(0.0, marginal_w1),
        equality_error=error,
    )


def causal_transport_stack(
    query: ArrayLike,
    kernels: Sequence[ArrayLike],
    sites: ArrayLike | None = None,
) -> CausalTransportStackResult:
    """Roll out causal kernels and enforce endpoint/layer W1 additivity."""
    initial = _probability(query, name="query")
    selected_sites = _ordered_sites(sites, size=initial.size)
    current = initial
    layers: list[CausalTransportResult] = []
    for kernel in kernels:
        result = causal_transport_w1(current, kernel, selected_sites)
        layers.append(result)
        current = result.output_marginal

    endpoint_w1 = _discrete_w1(initial, current, selected_sites)
    additive_w1 = math.fsum(layer.marginal_w1 for layer in layers)
    error = abs(endpoint_w1 - additive_w1)
    if error > _tolerance(endpoint_w1, additive_w1):
        raise ArithmeticError("causal stack violates the Foss geodesic identity")

    return CausalTransportStackResult(
        layers=tuple(layers),
        endpoint_w1=max(0.0, endpoint_w1),
        additive_w1=max(0.0, additive_w1),
        equality_error=error,
    )


def route_transport_w1(
    edges: Sequence[CausalEdge],
    target_names: Sequence[str],
) -> RouteTransportResult:
    """Measure normalized positive route-score transport per requested target.

    Positive scores for one target are normalized into probability mass and
    their lags are used as backward distances. Intervention edges therefore
    contribute zero because their lag is zero. Missing and zero-score routes
    have zero transport. The result is a transport statistic constructed from
    route scores, not a Wasserstein interpretation of raw correlations.
    """
    selected_edges = tuple(edges)
    if any(not isinstance(edge, CausalEdge) for edge in selected_edges):
        raise TypeError("edges must contain CausalEdge instances")

    requested: list[str] = []
    seen: set[str] = set()
    for target in target_names:
        if not isinstance(target, str) or not target:
            raise ValueError("target names must be non-empty strings")
        if target not in seen:
            seen.add(target)
            requested.append(target)

    per_target: dict[str, float] = {}
    for target in requested:
        positive = tuple(
            edge
            for edge in selected_edges
            if edge.target == target and edge.score > 0.0
        )
        total_score = math.fsum(edge.score for edge in positive)
        if total_score == 0.0:
            per_target[target] = 0.0
            continue
        per_target[target] = math.fsum(
            (edge.score / total_score) * float(edge.lag) for edge in positive
        )

    mean_w1 = math.fsum(per_target.values()) / len(requested) if requested else 0.0
    return RouteTransportResult(per_target_w1=per_target, mean_w1=mean_w1)
