"""Step 3: Dense + GraphExp, a non-learned graph-expansion reranker.

Given one query's retrieval scores over all clauses:

    d~  = min-max(d)                          in [0, 1]
    s   = d~ on the top-`seed_k` clauses, 0 elsewhere
    Â   = D^{-1/2} (A + I) D^{-1/2},  D = rowsum(A + I)
    out = d~ + lam * sum_{t=1..hops} Â^t s

Â is the GCN-normalized adjacency, so a seed spreads less score through a
high-degree hub than through a low-degree neighbor.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from clause_graph import DEFAULT_EDGE_TYPES, adjacency


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant vector maps to all zeros."""
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


def normalized_adjacency(a: np.ndarray) -> np.ndarray:
    """D^{-1/2} (A + I) D^{-1/2} for a symmetric 0/1 adjacency A."""
    a_hat = np.asarray(a, dtype=float) + np.eye(len(a))
    d = a_hat.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(d)
    return a_hat * inv_sqrt[:, None] * inv_sqrt[None, :]


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k highest scores, ties broken by lower index.

    Nodes are stored in sorted clause-ID order, so this is Erica's
    (score desc, clause ID asc) ordering.
    """
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:k]


def graph_expand(dense_scores: np.ndarray, graph: dict[str, Any] | np.ndarray, seed_k: int = 5,
                 hops: int = 1, lam: float = 0.5,
                 edge_types: tuple[str, ...] = DEFAULT_EDGE_TYPES) -> np.ndarray:
    """Spread normalized dense score from the top-`seed_k` clauses to graph neighbors.

    Args:
        dense_scores: shape [n_clauses], one query's scores in node order.
        graph: a clause graph dict (see clause_graph.py) or a precomputed
            normalized adjacency Â (shape [n, n]); with Â, `edge_types` is ignored.
        seed_k: number of top clauses that act as seeds.
        hops: number of propagation steps summed (1 = direct neighbors).
        lam: weight of the propagated score relative to the dense score.
        edge_types: edge types included in A.

    Returns:
        New scores, shape [n_clauses]. Rank by descending score.
    """
    d = minmax(dense_scores)
    a_norm = graph if isinstance(graph, np.ndarray) else normalized_adjacency(adjacency(graph, edge_types))
    if a_norm.shape != (len(d), len(d)):
        raise ValueError(f"Score vector has {len(d)} entries but the graph has {len(a_norm)} nodes")
    if seed_k < 1 or hops < 0:
        raise ValueError("seed_k must be >= 1 and hops >= 0")
    seeds = np.zeros_like(d)
    idx = top_k_indices(d, seed_k)
    seeds[idx] = d[idx]
    out, prop = d.copy(), seeds
    # Some numpy 2.2 builds on Apple BLAS raise spurious FP flags in matmul even
    # on finite inputs; ignore the flags and check the result instead.
    with np.errstate(all="ignore"):
        for _ in range(hops):
            prop = a_norm @ prop
            out += lam * prop
    if not np.isfinite(out).all():
        raise FloatingPointError("graph_expand produced non-finite scores")
    return out


def rank(scores: np.ndarray, ids: Sequence[str]) -> list[str]:
    """IDs by descending score, ties by position in `ids` (sorted clause IDs)."""
    return [ids[i] for i in top_k_indices(scores, len(ids))]
