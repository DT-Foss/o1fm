from __future__ import annotations

import json
from pathlib import Path

import pytest

from o1fm.causal_graph import (
    CausalGraphConfig,
    CausalGraphConflictError,
    CausalGraphIntegrityError,
    OnlineCausalGraph,
)


def _charge_correlated_graph(graph: OnlineCausalGraph, rows: int = 14) -> None:
    x = y = known = 0.0
    previous_dx = 0.0
    for timestamp in range(rows):
        dx = float((timestamp % 5) + 1)
        x += dx
        y += previous_dx
        known += previous_dx
        graph.observe(
            timestamp_ns=timestamp,
            targets={"demand": y},
            covariates={"driver": x},
            known_future={"promotion": known},
            observation_id=f"row-{timestamp}",
        )
        previous_dx = dx


def test_graph_learns_lagged_and_known_future_intervention_edges(
    tmp_path: Path,
) -> None:
    graph = OnlineCausalGraph(
        tmp_path / "graph",
        config=CausalGraphConfig(max_lag=2, min_observations=4, top_k=8),
    )
    _charge_correlated_graph(graph)

    edges = graph.all_edges(target="demand")
    lagged = next(
        edge
        for edge in edges
        if edge.source == "driver" and edge.lag == 1 and edge.kind == "observational"
    )
    intervention = next(
        edge
        for edge in edges
        if edge.source == "promotion" and edge.kind == "intervention"
    )
    assert lagged.score > 0.99
    assert intervention.score > 0.99
    assert lagged.observations >= 4
    assert {edge.kind for edge in graph.route("demand")} == {
        "intervention",
        "observational",
    }


def test_graph_revision_is_persistent_idempotent_and_top_k_bounded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "graph"
    graph = OnlineCausalGraph(
        root,
        graph_id="retail",
        config=CausalGraphConfig(max_lag=2, min_observations=3, top_k=1),
    )
    _charge_correlated_graph(graph, rows=10)
    before = graph.revision
    duplicate = graph.observe(
        timestamp_ns=9,
        targets={"demand": 25.0},
        covariates={"driver": 30.0},
        known_future={"promotion": 25.0},
        observation_id="row-9",
    )

    assert not duplicate.appended
    assert duplicate.sequence == before.sequence
    assert len(graph.route("demand")) == 1
    restored = OnlineCausalGraph(
        root,
        graph_id="retail",
        config=CausalGraphConfig(max_lag=2, min_observations=3, top_k=1),
    )
    assert restored.revision == graph.revision
    assert restored.all_edges() == graph.all_edges()

    with pytest.raises(CausalGraphConflictError):
        graph.observe(
            timestamp_ns=10,
            targets={"demand": 999.0},
            covariates={"driver": 30.0},
            known_future={"promotion": 20.0},
            observation_id="row-9",
        )


def test_graph_rejects_invalid_inputs_without_revision(tmp_path: Path) -> None:
    graph = OnlineCausalGraph(tmp_path / "graph")
    with pytest.raises(ValueError, match="disjoint"):
        graph.observe(
            timestamp_ns=1,
            targets={"x": 1.0},
            covariates={"x": 2.0},
        )
    with pytest.raises(ValueError, match="finite"):
        graph.observe(timestamp_ns=1, targets={"x": float("nan")})
    assert graph.revision.sequence == 0


def test_graph_hash_chain_detects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "graph"
    graph = OnlineCausalGraph(root)
    graph.observe(timestamp_ns=1, targets={"y": 1.0}, observation_id="one")
    event = next((root / "events").glob("*.json"))
    document = json.loads(event.read_text())
    document["body"]["timestamp_ns"] = 2
    event.chmod(0o600)
    event.write_text(json.dumps(document, separators=(",", ":"), sort_keys=True))

    with pytest.raises(CausalGraphIntegrityError):
        OnlineCausalGraph(root)
