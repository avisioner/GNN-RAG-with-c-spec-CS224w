"""Step 2: build the 82-node clause graph from chunks.jsonl.

Edge types (all between corpus clauses, self-loops removed):
  references          A cites B in any of A's chunks (anchors like `#nt:...` stripped). Directed.
  parent              A's parent_clause is corpus clause B. Empty on the current data.
  sibling             A and B share a parent_clause. Undirected, stored with source < target.
  references_plain    subset of `references` from un-anchored mentions only; equals what
                      spec_graph.json contains. Variant only.
  references_general  A cites an out-of-corpus section T whose text is corpus clause `T.general`.
                      Variant only.

    python graph/specgraph_graph/clause_graph.py      # writes artifacts/graph/clause_graph.json

Teammates: `from clause_graph import load_clause_graph`.
"""
from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common import CLAUSE_GRAPH_PATH, corpus_clauses, load_chunks, norm_id, write_json

EDGE_TYPES = ("references", "parent", "sibling", "references_plain", "references_general")
UNDIRECTED_TYPES = {"sibling"}
DEFAULT_EDGE_TYPES = ("references", "parent", "sibling")


def build_clause_graph(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return {nodes, edges, dropped} built from chunk records.

    Nodes are sorted clause IDs; their order defines the index used by every
    adjacency matrix and score vector.
    """
    clauses = corpus_clauses(chunks)
    corpus = set(clauses)
    meta: dict[str, dict[str, Any]] = {}
    for c in chunks:
        m = meta.setdefault(c["clause_id"], {"clause_id": c["clause_id"], "title": c.get("title", ""),
                                            "section_id": c.get("section_id"),
                                            "parent_clause": c.get("parent_clause"), "n_chunks": 0})
        m["n_chunks"] += 1

    edges: dict[str, set[tuple[str, str]]] = {t: set() for t in EDGE_TYPES}
    dropped = Counter()
    dropped_targets = Counter()
    for c in chunks:
        src = c["clause_id"]
        for ref in c.get("cross_references", []):
            raw = ref["target_clause_id"]
            tgt = norm_id(raw)
            if tgt == src:
                dropped["self_loop"] += 1
            elif tgt in corpus:
                edges["references"].add((src, tgt))
                if "#" not in raw:
                    edges["references_plain"].add((src, tgt))
            else:
                dropped["out_of_corpus"] += 1
                dropped_targets[tgt] += 1
                general = f"{tgt}.general"
                if general in corpus and general != src:
                    edges["references_general"].add((src, general))

    for cid in clauses:
        parent = meta[cid]["parent_clause"]
        if parent in corpus and parent != cid:
            edges["parent"].add((cid, parent))

    groups: dict[str, list[str]] = {}
    for cid in clauses:
        groups.setdefault(meta[cid]["parent_clause"], []).append(cid)
    for members in groups.values():
        edges["sibling"].update(combinations(sorted(members), 2))

    return {
        "nodes": [meta[c] for c in clauses],
        "edges": [{"source": s, "target": t, "edge_type": et}
                  for et in EDGE_TYPES for s, t in sorted(edges[et])],
        "dropped": {"mentions": dict(dropped), "top_out_of_corpus_targets": dropped_targets.most_common(20),
                    "distinct_out_of_corpus_targets": len(dropped_targets)},
    }


def node_ids(graph: dict[str, Any]) -> list[str]:
    return [n["clause_id"] for n in graph["nodes"]]


def adjacency(graph: dict[str, Any], edge_types: Iterable[str] = DEFAULT_EDGE_TYPES) -> np.ndarray:
    """Symmetric 0/1 adjacency (no self-loops) over the chosen edge types."""
    types = set(edge_types)
    unknown = types - set(EDGE_TYPES)
    if unknown:
        raise ValueError(f"Unknown edge types: {sorted(unknown)}")
    index = {c: i for i, c in enumerate(node_ids(graph))}
    a = np.zeros((len(index), len(index)))
    for e in graph["edges"]:
        if e["edge_type"] in types:
            i, j = index[e["source"]], index[e["target"]]
            a[i, j] = a[j, i] = 1.0
    np.fill_diagonal(a, 0.0)
    return a


def _components(a: np.ndarray) -> list[int]:
    n, seen, sizes = len(a), set(), []
    for s in range(n):
        if s in seen:
            continue
        seen.add(s)
        queue, size = deque([s]), 0
        while queue:
            u = queue.popleft()
            size += 1
            for v in np.flatnonzero(a[u]):
                if v not in seen:
                    seen.add(int(v))
                    queue.append(int(v))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def hop_share(a: np.ndarray, hops: int) -> float:
    """Mean over clauses of |{v} ∪ N_<=hops(v)| / n."""
    n = len(a)
    reach = np.eye(n, dtype=bool)
    step = a.astype(bool)
    for _ in range(hops):
        reach = reach | ((reach.astype(int) @ step.astype(int)) > 0)
    return float(reach.sum(axis=1).mean() / n)


def graph_stats(graph: dict[str, Any]) -> dict[str, Any]:
    n = len(graph["nodes"])
    by_type = Counter(e["edge_type"] for e in graph["edges"])
    out: dict[str, Any] = {"nodes": n, "edges_stored": dict(by_type), "undirected_edges": {}, "configs": {}}
    for et in EDGE_TYPES:
        out["undirected_edges"][et] = int(adjacency(graph, [et]).sum() // 2)
    configs = {"default (references+parent+sibling)": DEFAULT_EDGE_TYPES, "references": ("references",),
               "sibling": ("sibling",), "parent": ("parent",), "references_plain": ("references_plain",),
               "references+references_general+sibling": ("references", "references_general", "sibling")}
    ids = node_ids(graph)
    for name, types in configs.items():
        a = adjacency(graph, types)
        deg = a.sum(axis=1).astype(int)
        comps = _components(a)
        out["configs"][name] = {
            "undirected_edges": int(a.sum() // 2),
            "degree": {"min": int(deg.min()), "median": float(np.median(deg)), "mean": round(float(deg.mean()), 2),
                       "max": int(deg.max()), "max_node": ids[int(deg.argmax())]},
            "degree_histogram": {str(k): v for k, v in sorted(Counter(deg.tolist()).items())},
            "isolated": [ids[i] for i in np.flatnonzero(deg == 0)],
            "components": len(comps), "largest_component": comps[0],
            "component_sizes": comps[:10],
            "mean_share_within_1_hop": round(hop_share(a, 1), 4),
            "mean_share_within_2_hops": round(hop_share(a, 2), 4),
        }
    out["dropped"] = graph["dropped"]
    return out


def save_clause_graph(graph: dict[str, Any], path: Path = CLAUSE_GRAPH_PATH) -> None:
    write_json(path, {"version": "milestone-1", "source": "data/processed/chunks.jsonl",
                      "node_order": "sorted clause_id", "edge_types": list(EDGE_TYPES),
                      "undirected_edge_types": sorted(UNDIRECTED_TYPES), "default_edge_types": list(DEFAULT_EDGE_TYPES),
                      **graph, "stats": graph_stats(graph)})


def load_clause_graph(path: Path = CLAUSE_GRAPH_PATH) -> dict[str, Any]:
    """Load the saved clause graph: {nodes, edges, stats, ...}.

    Use `node_ids(graph)` for the node order and `adjacency(graph, edge_types)`
    for a symmetric numpy adjacency matrix.
    """
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    g = build_clause_graph(load_chunks())
    save_clause_graph(g)
    s = graph_stats(g)
    print(f"Wrote {CLAUSE_GRAPH_PATH}")
    print("undirected edges:", s["undirected_edges"])
    for name, c in s["configs"].items():
        print(f"{name}: edges={c['undirected_edges']} deg={c['degree']} isolated={len(c['isolated'])} "
              f"components={c['components']} 1hop={c['mean_share_within_1_hop']} 2hop={c['mean_share_within_2_hops']}")
