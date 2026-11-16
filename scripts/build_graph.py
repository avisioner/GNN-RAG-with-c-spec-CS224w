#!/usr/bin/env python3
"""Build the specification graph from parsed clause chunks.

Reads data/processed/chunks.jsonl and writes data/graph/spec_graph.json.
Running the script again replaces that file with a new mapping.

Each stored clause becomes a clause node. A parent_clause that has no text
of its own, such as expr.compound, becomes a container node. Reference edges
connect two nodes that both exist in this graph. Links to clauses that have
not been parsed, and links to grammar productions, are left out.

Example:
    python scripts/build_graph.py
    python scripts/build_graph.py -i data/processed/chunks.jsonl -o data/graph/spec_graph.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from parse_core_spec import load_chunks

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_GRAPH = ROOT / "data" / "graph" / "spec_graph.json"


def clause_target(target: str) -> str:
    """Return a clause id, or an empty string when the link is not a clause."""
    if not target or "#" in target:
        return ""
    return target


def anchor_titles(chunks: list[dict[str, object]]) -> dict[str, str]:
    """Best anchor text seen for each cross-reference target."""
    titles: dict[str, str] = {}
    for chunk in chunks:
        for reference in chunk.get("cross_references") or []:
            if not isinstance(reference, dict):
                continue
            target = clause_target(str(reference.get("target_clause_id") or ""))
            title = str(reference.get("anchor_text") or "").strip()
            if target and title and len(title) > len(titles.get(target, "")):
                titles[target] = title
    return titles


def build_graph(chunks: list[dict[str, object]]) -> dict[str, object]:
    by_clause: dict[str, list[dict[str, object]]] = defaultdict(list)
    for chunk in chunks:
        clause_id = str(chunk.get("clause_id") or "")
        if clause_id:
            by_clause[clause_id].append(chunk)
    if not by_clause:
        raise SystemExit("No clause chunks to build a graph from.")

    titles = anchor_titles(chunks)
    editions = {str(chunk.get("spec_edition") or "") for chunk in chunks}
    editions.discard("")
    edition = sorted(editions)[0] if len(editions) == 1 else ""

    nodes: list[dict[str, object]] = []
    for clause_id in sorted(by_clause):
        group = by_clause[clause_id]
        chunk_id = str(group[0].get("chunk_id") or "") if len(group) == 1 else None
        nodes.append(
            {
                "node_id": f"clause:{clause_id}",
                "node_type": "clause",
                "clause_id": clause_id,
                "title": group[0].get("title") or titles.get(clause_id) or "",
                "chunk_id": chunk_id,
            }
        )

    parents = {
        str(group[0].get("parent_clause") or "")
        for group in by_clause.values()
    }
    parents.discard("")
    for clause_id in sorted(parents - set(by_clause)):
        nodes.append(
            {
                "node_id": f"clause:{clause_id}",
                "node_type": "container",
                "clause_id": clause_id,
                "title": titles.get(clause_id) or "",
                "chunk_id": None,
            }
        )

    node_ids = {str(node["clause_id"]) for node in nodes}
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_edge(source: str, target: str, edge_type: str) -> None:
        if not source or not target or source == target:
            return
        if source not in node_ids or target not in node_ids:
            return
        key = (source, target, edge_type)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "source": f"clause:{source}",
                "target": f"clause:{target}",
                "edge_type": edge_type,
                "provenance": "automatic",
            }
        )

    for clause_id, group in by_clause.items():
        add_edge(clause_id, str(group[0].get("parent_clause") or ""), "parent")
        for chunk in group:
            for reference in chunk.get("cross_references") or []:
                if not isinstance(reference, dict):
                    continue
                target = clause_target(str(reference.get("target_clause_id") or ""))
                edge_type = str(reference.get("edge_type") or "references")
                add_edge(clause_id, target, edge_type)

    edges.sort(key=lambda edge: (edge["edge_type"], edge["source"], edge["target"]))
    nodes.sort(key=lambda node: (str(node["node_type"]), str(node["node_id"])))
    return {
        "graph_version": "1.0",
        "spec_edition": edition,
        "nodes": nodes,
        "edges": edges,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build spec_graph.json from chunks.jsonl.")
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_CHUNKS, help="Chunk file.")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_GRAPH, help="Graph file.")
    args = parser.parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Chunk file not found: {args.input}")

    graph = build_graph(load_chunks(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    clause_nodes = sum(1 for node in graph["nodes"] if node["node_type"] == "clause")
    container_nodes = sum(1 for node in graph["nodes"] if node["node_type"] == "container")
    references = sum(1 for edge in graph["edges"] if edge["edge_type"] == "references")
    parents = sum(1 for edge in graph["edges"] if edge["edge_type"] == "parent")
    sys.stdout.write(
        f"Wrote {args.output}: {clause_nodes} clauses, {container_nodes} containers, "
        f"{references} reference edges, {parents} parent edges.\n"
    )


if __name__ == "__main__":
    main()
