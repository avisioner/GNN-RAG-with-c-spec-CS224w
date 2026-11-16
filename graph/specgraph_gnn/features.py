"""Query-conditioned node features and relation edge lists for the clause graph.

Features (7 per clause, no clause-identity features):
    0  dense max-chunk score, z-scored within the query
    1  dense mean-chunk score, z-scored within the query
    2  BM25 max-chunk score, z-scored within the query
    3  1 / rank under dense (max-chunk), ties by clause ID
    4  1 / rank under BM25, ties by clause ID
    5  1 if in the dense top-5
    6  log(1 + degree), standardized over clauses (same for every query)
The brief's optional constant-1 slot is dropped: every layer already has a bias.

Relations (brief §7, edge types kept exactly as named in clause_graph.json):
    references       source -> target (the citing clause sends to the cited one)
    references (rev) target -> source (same edge type, opposite direction)
    sibling          both directions (stored undirected)
`parent` is in default_edge_types but has no edges on this corpus (no parent
section is a corpus clause); build_relations asserts that rather than hiding it.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from gnn_common import graph_common  # noqa: F401  (sets sys.path for the graph module)

from clause_graph import adjacency, load_clause_graph, node_ids  # noqa: E402

N_FEATURES = 7
TOP_K_FLAG = 5
EPS = 1e-8

# name -> list of (edge_type, direction); direction "fwd", "rev" or "both".
RELATION_SETS: dict[str, list[tuple[str, str]]] = {
    "refs+sibling": [("references", "fwd"), ("references", "rev"), ("sibling", "both")],
    "refs only": [("references", "fwd"), ("references", "rev")],
}


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardize each row (one query) to mean 0, std 1; a constant row maps to 0."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / np.maximum(sd, EPS)


def reciprocal_rank(x: np.ndarray) -> np.ndarray:
    """1/rank per row, rank 1 = highest score, ties broken by lower column index (clause ID asc)."""
    n_q, n = x.shape
    out = np.zeros_like(x, dtype=float)
    for i in range(n_q):
        order = sorted(range(n), key=lambda j: (-float(x[i, j]), j))
        out[i, order] = 1.0 / np.arange(1, n + 1)
    return out


def log_degree_feature(adj: np.ndarray) -> np.ndarray:
    d = np.log1p(adj.sum(axis=1))
    return (d - d.mean()) / max(d.std(), EPS)


def build_features(scores: dict[str, np.ndarray], degree_feat: np.ndarray) -> np.ndarray:
    """[n_queries, n_clauses, N_FEATURES] float32 array."""
    dmax, dmean, bm = scores["dense_max"], scores["dense_mean"], scores["bm25_max"]
    if not (dmax.shape == dmean.shape == bm.shape) or dmax.shape[1] != len(degree_feat):
        raise ValueError("Score arrays and degree feature disagree on shape")
    rr_dense = reciprocal_rank(dmax)
    top = (rr_dense >= 1.0 / TOP_K_FLAG).astype(float)
    feats = [zscore(dmax), zscore(dmean), zscore(bm), rr_dense, reciprocal_rank(bm), top,
             np.broadcast_to(degree_feat, dmax.shape)]
    return np.stack(feats, axis=-1).astype(np.float32)


def build_relations(graph: dict[str, Any], relation_set: Sequence[tuple[str, str]]
                    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(edge_index [2, E], edge_type [E], relation names) in PyG source->target convention."""
    index = {c: i for i, c in enumerate(node_ids(graph))}
    if any(e["edge_type"] == "parent" for e in graph["edges"]):
        raise ValueError("clause_graph has parent edges now; add them to RELATION_SETS deliberately")
    src: list[int] = []
    dst: list[int] = []
    rel: list[int] = []
    names: list[str] = []
    for r, (etype, direction) in enumerate(relation_set):
        names.append(f"{etype} ({direction})")
        pairs = {(index[e["source"]], index[e["target"]]) for e in graph["edges"] if e["edge_type"] == etype}
        if not pairs:
            raise ValueError(f"No edges of type {etype!r}")
        if direction == "rev":
            pairs = {(j, i) for i, j in pairs}
        elif direction == "both":
            pairs |= {(j, i) for i, j in pairs}
        for i, j in sorted(pairs):
            if i != j:
                src.append(i)
                dst.append(j)
                rel.append(r)
    return np.array([src, dst], dtype=np.int64), np.array(rel, dtype=np.int64), names


def edge_types_of(relation_set: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(t for t, _ in relation_set))


def degree_feature_for(graph: dict[str, Any], relation_set: Sequence[tuple[str, str]]) -> np.ndarray:
    """Degree over the same edge types the model sees, so the no-sibling ablation drops sibling info entirely."""
    return log_degree_feature(adjacency(graph, edge_types_of(relation_set)))


__all__ = ["N_FEATURES", "RELATION_SETS", "build_features", "build_relations", "degree_feature_for",
           "load_clause_graph", "node_ids", "reciprocal_rank", "zscore"]
