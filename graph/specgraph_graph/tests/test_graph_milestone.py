"""Hand-checked tests for graph expansion, metrics, ID normalization and graph building.

    python -m pytest graph/specgraph_graph/tests -q
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clause_graph import adjacency, build_clause_graph  # noqa: E402
from common import norm_id, normalize_difficulty  # noqa: E402
from evaluate import boot_ci, paired, per_query_metrics, reachability  # noqa: E402
from graph_expand import graph_expand, minmax, normalized_adjacency, rank  # noqa: E402

R6 = math.sqrt(6)

# Path a - b - c, plus isolated d.  With self-loops, degrees are (2, 3, 2, 1).
TINY = {"nodes": [{"clause_id": c} for c in "abcd"],
        "edges": [{"source": "a", "target": "b", "edge_type": "references"},
                  {"source": "c", "target": "b", "edge_type": "sibling"}]}
DENSE = np.array([4.0, 0.0, 2.0, 1.0])          # min-max -> (1, 0, 0.5, 0.25)


def test_normalized_adjacency_by_hand():
    expected = np.array([[1 / 2, 1 / R6, 0, 0],
                         [1 / R6, 1 / 3, 1 / R6, 0],
                         [0, 1 / R6, 1 / 2, 0],
                         [0, 0, 0, 1]])
    np.testing.assert_allclose(normalized_adjacency(adjacency(TINY, ("references", "sibling"))), expected)


def test_graph_expand_one_hop_by_hand():
    # Seed = a only (score 1).  Â s = (1/2, 1/√6, 0, 0).
    out = graph_expand(DENSE, TINY, seed_k=1, hops=1, lam=0.5, edge_types=("references", "sibling"))
    np.testing.assert_allclose(out, [1 + 0.25, 0.5 / R6, 0.5, 0.25])


def test_graph_expand_two_hops_by_hand():
    # Â² s = Â (1/2, 1/√6, 0, 0) = (1/4 + 1/6, (1/2 + 1/3)/√6, 1/6, 0).
    out = graph_expand(DENSE, TINY, seed_k=1, hops=2, lam=0.5, edge_types=("references", "sibling"))
    a1 = np.array([1 / 2, 1 / R6, 0, 0])
    a2 = np.array([5 / 12, (5 / 6) / R6, 1 / 6, 0])
    np.testing.assert_allclose(out, minmax(DENSE) + 0.5 * (a1 + a2))


def test_graph_expand_two_seeds_and_edge_filter():
    # Seeds a (1.0) and c (0.5); references only: a - b, c isolated (degree 1).
    out = graph_expand(DENSE, TINY, seed_k=2, hops=1, lam=1.0, edge_types=("references",))
    # Â over {a-b}: a,b degree 2; c,d degree 1.  Â s = (1/2, 1/2, 0.5, 0).
    np.testing.assert_allclose(out, [1.5, 0.5, 1.0, 0.25])


def test_graph_expand_identities():
    assert np.allclose(graph_expand(DENSE, TINY, lam=0.0), minmax(DENSE))
    assert np.allclose(graph_expand(DENSE, TINY, hops=0), minmax(DENSE))
    empty = {"nodes": TINY["nodes"], "edges": []}
    # No edges: Â = I, so the seeds just get (1 + λ) times their score.
    out = graph_expand(DENSE, empty, seed_k=1, lam=0.5)
    np.testing.assert_allclose(out, [1.5, 0, 0.5, 0.25])
    with pytest.raises(ValueError):
        graph_expand(np.zeros(3), TINY)


def test_graph_expand_accepts_precomputed_adjacency():
    a_norm = normalized_adjacency(adjacency(TINY, ("references", "sibling")))
    np.testing.assert_allclose(graph_expand(DENSE, a_norm, seed_k=1),
                               graph_expand(DENSE, TINY, seed_k=1, edge_types=("references", "sibling")))


def test_rank_breaks_ties_by_id_order():
    assert rank(np.array([1.0, 2.0, 1.0, 2.0]), list("abcd")) == ["b", "d", "a", "c"]


def test_metrics_hand_example():
    m = per_query_metrics(["a", "x", "b", "y", "c"], ["a", "b", "c"], ks=(1, 3, 5))
    assert m["recall@1"] == pytest.approx(1 / 3) and m["precision@1"] == 1.0
    assert m["recall@3"] == pytest.approx(2 / 3) and m["precision@3"] == pytest.approx(2 / 3)
    assert m["complete_coverage@3"] == 0.0
    assert m["recall@5"] == 1.0 and m["precision@5"] == pytest.approx(3 / 5)
    assert m["complete_coverage@5"] == 1.0
    # nDCG@3: hits at ranks 1 and 3; ideal has 3 hits in the first 3.
    dcg = 1 + 1 / math.log2(4)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m["ndcg@3"] == pytest.approx(dcg / ideal)


def test_metrics_reject_duplicate_ranking():
    with pytest.raises(ValueError):
        per_query_metrics(["a", "a"], ["a"], ks=(1,))


def test_boot_ci_and_paired():
    mean, lo, hi = boot_ci([1, 0, 1, 1])
    assert mean == 0.75 and 0 <= lo <= mean <= hi <= 1
    assert boot_ci([1, 0, 1, 1]) == (mean, lo, hi)          # fixed seed
    assert boot_ci([1, 1, 1]) == (1.0, 1.0, 1.0)
    p = paired([1, 0, 1, 0], [0, 0, 1, 1])
    assert (p["wins"], p["losses"], p["ties"]) == (1, 1, 2) and p["mean_diff"] == 0.0


def test_reachability_by_hand():
    adj = adjacency(TINY, ("references", "sibling"))
    r = reachability(["a"], ["a", "c", "d"], adj, list("abcd"))
    # seeds {a}, N = {b}; reach {a, b}.  Gold a is reached, c and d are not.
    assert r["gold_in_reach"] == pytest.approx(1 / 3) and r["chance_reach"] == 0.5
    assert r["missed_found"] == 0.0 and r["chance_missed"] == pytest.approx(1 / 3) and r["n_missed"] == 2


@pytest.mark.parametrize("raw, expected", [
    ("clause:expr.pre#p2", "expr.pre"), ("expr.pre", "expr.pre"), ("clause:expr.post.incr", "expr.post.incr"),
    ("dcl.init.general#nt:initializer", "dcl.init.general"), ("  clause:conv#def:x ", "conv"),
])
def test_norm_id(raw, expected):
    assert norm_id(raw) == expected


def test_norm_id_rejects_empty():
    with pytest.raises(ValueError):
        norm_id("")


@pytest.mark.parametrize("raw, expected", [
    ("multi_hop", "multi_hop"), ("multi clause", "multi_hop"), ("Multi-Hop", "multi_hop"),
    ("multi_clause", "multi_hop"), ("single_clause", "single_clause"), ("two clause", "two_clause"),
])
def test_normalize_difficulty(raw, expected):
    assert normalize_difficulty(raw) == expected


def test_normalize_difficulty_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_difficulty("hard")


def _chunk(cid, parent, refs, suffix=""):
    return {"chunk_id": f"clause:{cid}{suffix}", "clause_id": cid, "parent_clause": parent, "title": cid,
            "cross_references": [{"target_clause_id": t, "edge_type": "references"} for t in refs]}


def test_build_clause_graph_rules():
    chunks = [
        _chunk("x.a", "x", ["x.b#nt:foo", "x.a", "outside", "sec"], "#p1"),
        _chunk("x.a", "x", ["x.b"], "#p2"),                       # duplicate edge across chunks
        _chunk("x.b", "x", []),
        _chunk("y.c", "x.a", ["sec"]),                            # parent is a corpus clause
        _chunk("sec.general", "sec", []),
    ]
    g = build_clause_graph(chunks)
    edges = {(e["source"], e["target"], e["edge_type"]) for e in g["edges"]}
    assert ("x.a", "x.b", "references") in edges
    assert ("x.a", "x.b", "references_plain") in edges                 # from the un-anchored #p2 mention
    assert not any(e["source"] == e["target"] for e in g["edges"])       # self-loop removed
    assert ("x.a", "x.b", "sibling") in edges
    assert ("y.c", "x.a", "parent") in edges
    assert ("x.a", "sec.general", "references_general") in edges
    assert ("y.c", "sec.general", "references_general") in edges
    assert sum(e["edge_type"] == "references" for e in g["edges"]) == 1  # deduplicated
    assert g["dropped"]["mentions"] == {"self_loop": 1, "out_of_corpus": 3}
