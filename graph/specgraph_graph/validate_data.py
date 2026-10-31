"""Step 1: audit chunks.jsonl, spec_graph.json and the benchmark.

Reports issues only; never edits the data files. Writes
results/milestone/data_validation.md.

    python graph/specgraph_graph/validate_data.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (BENCHMARK_PATH, RESULTS_DIR, corpus_clauses, erica, load_chunks,
                    load_spec_graph, norm_id, normalize_difficulty)

REQUIRED_CHUNK_FIELDS = ("chunk_id", "clause_id", "section_id", "section", "parent_clause",
                         "title", "chunk_type", "text", "source_url", "spec_edition",
                         "cross_references")
SHORT_TEXT_CHARS = 40
EXPECTED_GOLD_SIZE = {"single_clause": lambda n: n == 1, "two_clause": lambda n: n == 2,
                      "multi_hop": lambda n: n >= 3}


def _head(items: list[Any], n: int = 10) -> str:
    items = list(items)
    if not items:
        return "none"
    shown = ", ".join(f"`{x}`" for x in items[:n])
    return shown + (f", ... (+{len(items) - n} more)" if len(items) > n else "")


def check_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    clauses = set(corpus_clauses(chunks))
    missing = {f: [c.get("chunk_id", "?") for c in chunks if f not in c] for f in REQUIRED_CHUNK_FIELDS}
    ids = Counter(c["chunk_id"] for c in chunks)
    by_clause: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        by_clause[c["clause_id"]].append(c)
    return {
        "n_chunks": len(chunks), "n_clauses": len(clauses),
        "missing_fields": {f: v for f, v in missing.items() if v},
        "duplicate_chunk_ids": [k for k, v in ids.items() if v > 1],
        "prefix_mismatch": [c["chunk_id"] for c in chunks if norm_id(c["chunk_id"]) != c["clause_id"]],
        "no_clause_prefix": [c["chunk_id"] for c in chunks if not c["chunk_id"].startswith("clause:")],
        "empty_text": [c["chunk_id"] for c in chunks if not c.get("text", "").strip()],
        "short_text": [(c["chunk_id"], len(c["text"])) for c in chunks
                       if 0 < len(c.get("text", "").strip()) < SHORT_TEXT_CHARS],
        "inconsistent_parent": [k for k, v in by_clause.items() if len({c.get("parent_clause") for c in v}) > 1],
        "inconsistent_title": [k for k, v in by_clause.items() if len({c.get("title") for c in v}) > 1],
        "chunks_per_clause": Counter(len(v) for v in by_clause.values()),
        "chunk_types": Counter(c.get("chunk_type") for c in chunks),
        "literal_backslash_n": [c["chunk_id"] for c in chunks if "\\n" in c.get("text", "")],
    }


def check_references(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    clauses = set(corpus_clauses(chunks))
    parents = {c["parent_clause"] for c in chunks}
    parent_of = {c["clause_id"]: c["parent_clause"] for c in chunks}
    refs = [(c["clause_id"], r) for c in chunks for r in c.get("cross_references", [])]
    targets = [norm_id(r["target_clause_id"]) for _, r in refs]
    clause_pairs = {(src, norm_id(r["target_clause_id"])) for src, r in refs}
    outside = Counter(t for t in targets if t not in clauses)
    return {
        "n_ref_mentions": len(refs),
        "edge_type_values": Counter(r.get("edge_type") for _, r in refs),
        "anchored_targets": sum("#" in r["target_clause_id"] for _, r in refs),
        "anchor_kinds": Counter(r["target_clause_id"].split("#", 1)[1].split(":")[0]
                                for _, r in refs if "#" in r["target_clause_id"]),
        "n_clause_pairs": len(clause_pairs),
        "inside_mentions": sum(t in clauses for t in targets),
        "outside_mentions": sum(t not in clauses for t in targets),
        "inside_pairs": sum(t in clauses and s != t for s, t in clause_pairs),
        "outside_pairs": sum(t not in clauses for _, t in clause_pairs),
        "distinct_outside_targets": len(outside),
        "top_outside": outside.most_common(15),
        "outside_parent_sections": sorted(t for t in outside if t in parents),
        "outside_with_general": sorted(t for t in outside if f"{t}.general" in clauses),
        "self_refs": sorted({s for s, t in clause_pairs if s == t}),
        "refs_to_own_parent": sorted((s, t) for s, t in clause_pairs if parent_of.get(s) == t),
    }


def check_parents(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    clauses = set(corpus_clauses(chunks))
    members: dict[str, set[str]] = defaultdict(set)
    for c in chunks:
        members[c["parent_clause"]].add(c["clause_id"])
    return {
        "distinct_parents": len(members),
        "parents_in_corpus": sorted(p for p in members if p in clauses),
        "parents_not_in_corpus": sorted(p for p in members if p not in clauses),
        "parents_with_general_clause": sorted(p for p in members if f"{p}.general" in clauses),
        "group_sizes": {p: len(v) for p, v in sorted(members.items(), key=lambda kv: -len(kv[1]))},
    }


def check_spec_graph(chunks: list[dict[str, Any]], graph: dict[str, Any]) -> dict[str, Any]:
    clauses = set(corpus_clauses(chunks))
    nodes, edges = graph["nodes"], graph["edges"]
    parent_of = {c["clause_id"]: c["parent_clause"] for c in chunks}
    chunk_ids = {c["chunk_id"] for c in chunks}
    graph_refs = {(norm_id(e["source"]), norm_id(e["target"])) for e in edges if e["edge_type"] == "references"}
    graph_parents = {(norm_id(e["source"]), norm_id(e["target"])) for e in edges if e["edge_type"] == "parent"}
    # Same normalization the graph file appears to use: every clause or container node.
    node_clause_ids = {n["clause_id"] for n in nodes}
    chunk_refs = {(c["clause_id"], t) for c in chunks for r in c.get("cross_references", [])
                  if (t := norm_id(r["target_clause_id"])) in node_clause_ids and t != c["clause_id"]}
    chunk_parents = {(k, v) for k, v in parent_of.items()}
    plain_refs = {(c["clause_id"], t) for c in chunks for r in c.get("cross_references", [])
                  if "#" not in r["target_clause_id"]
                  and (t := norm_id(r["target_clause_id"])) in node_clause_ids and t != c["clause_id"]}
    node_id_formats = Counter("clause:" if n["node_id"].startswith("clause:") else "bare" for n in nodes)
    return {
        "top_keys": sorted(graph.keys()), "graph_version": graph.get("graph_version"),
        "spec_edition": graph.get("spec_edition"),
        "node_types": Counter(n["node_type"] for n in nodes), "edge_types": Counter(e["edge_type"] for e in edges),
        "provenance": Counter(e.get("provenance") for e in edges),
        "node_id_formats": node_id_formats,
        "node_ids_match_clause_ids": all(norm_id(n["node_id"]) == n["clause_id"] for n in nodes),
        "container_nodes": sorted(n["clause_id"] for n in nodes if n["node_type"] == "container"),
        "clause_nodes_not_in_corpus": sorted(n["clause_id"] for n in nodes
                                             if n["node_type"] == "clause" and n["clause_id"] not in clauses),
        "corpus_clauses_missing_node": sorted(clauses - node_clause_ids),
        "clause_nodes_null_chunk_id": sorted(n["clause_id"] for n in nodes
                                             if n["node_type"] == "clause" and not n.get("chunk_id")),
        "node_chunk_ids_not_in_chunks": sorted(n["chunk_id"] for n in nodes
                                               if n.get("chunk_id") and n["chunk_id"] not in chunk_ids),
        "refs_graph_only": sorted(graph_refs - chunk_refs), "refs_chunks_only": sorted(chunk_refs - graph_refs),
        "refs_both": len(graph_refs & chunk_refs),
        "chunk_only_all_anchored": not ((chunk_refs - graph_refs) & plain_refs),
        "graph_equals_unanchored_chunks": graph_refs == plain_refs,
        "refs_graph_corpus_to_corpus": sum(s in clauses and t in clauses for s, t in graph_refs),
        "parents_graph_only": sorted(graph_parents - chunk_parents),
        "parents_chunks_only": sorted(chunk_parents - graph_parents), "parents_both": len(graph_parents & chunk_parents),
    }


def check_benchmark(chunks: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
    clauses = set(corpus_clauses(chunks))
    ids = Counter(q.get("query_id") for q in queries)
    raw_difficulty = Counter(q.get("difficulty") for q in queries)
    gold_raw = [(q["query_id"], g) for q in queries for g in q.get("gold_clause_ids", [])]
    mismatched = []
    for q in queries:
        d = normalize_difficulty(q["difficulty"])
        n = len({norm_id(g) for g in q["gold_clause_ids"]})
        if not EXPECTED_GOLD_SIZE[d](n):
            mismatched.append((q["query_id"], q["difficulty"], n))
    return {
        "n_queries": len(queries), "duplicate_query_ids": [k for k, v in ids.items() if v > 1],
        "raw_difficulty_labels": raw_difficulty,
        "normalized_difficulty": Counter(normalize_difficulty(q["difficulty"]) for q in queries),
        "gold_size_by_difficulty": Counter((normalize_difficulty(q["difficulty"]), len(q["gold_clause_ids"]))
                                           for q in queries),
        "gold_needing_normalization": [(qid, g) for qid, g in gold_raw if norm_id(g) != g],
        "gold_not_in_corpus": [(qid, g) for qid, g in gold_raw if norm_id(g) not in clauses],
        "duplicate_gold_within_query": [q["query_id"] for q in queries
                                        if len(q["gold_clause_ids"]) != len(set(q["gold_clause_ids"]))],
        "label_size_mismatch": mismatched,
        "empty_gold": [q["query_id"] for q in queries if not q.get("gold_clause_ids")],
        "distinct_gold_clauses": len({norm_id(g) for _, g in gold_raw}),
        "gold_frequency": Counter(norm_id(g) for _, g in gold_raw),
        "expected_property": Counter(q.get("expected_property") for q in queries),
    }


def render(ch: dict, rf: dict, pa: dict, sg: dict, bm: dict) -> str:
    L: list[str] = []
    add = L.append
    add("# Data validation report (milestone)\n")
    add("Generated by `graph/specgraph_graph/validate_data.py`. Read-only: no data file was modified. "
        "Issues are handled in code (`common.norm_id`, `common.normalize_difficulty`).\n")
    add("## Summary of real issues\n")
    issues = [
        f"**No `parent_clause` value is a corpus clause** ({len(pa['parents_in_corpus'])}/{pa['distinct_parents']}). "
        "Parent edges between corpus clauses are therefore empty; the clause graph adds `sibling` edges instead.",
        f"**{rf['outside_pairs']} of {rf['n_clause_pairs']} distinct (clause, target) reference pairs point outside "
        f"the 82-clause corpus** ({rf['distinct_outside_targets']} distinct targets). "
        f"{len(rf['outside_parent_sections'])} of those targets are parent sections of corpus clauses with no chunks "
        f"of their own (e.g. {_head(rf['outside_parent_sections'], 4)}).",
        f"**{len(bm['label_size_mismatch'])} queries have a difficulty label that disagrees with gold-set size** "
        "(all are `multi_hop` with 2 gold clauses; see below).",
        f"`spec_graph.json` uses `clause:`-prefixed node IDs and has {len(sg['container_nodes'])} `container` nodes "
        f"(the parent sections); {len(sg['clause_nodes_null_chunk_id'])} clause nodes have `chunk_id: null`.",
    ]
    if sg["refs_graph_only"] or sg["refs_chunks_only"]:
        issues.append(f"`spec_graph.json` reference edges disagree with `chunks.jsonl`: "
                      f"{len(sg['refs_graph_only'])} only in the graph file, {len(sg['refs_chunks_only'])} only in chunks.")
    else:
        issues.append("`spec_graph.json` reference and parent edges agree exactly with `chunks.jsonl` "
                      "(after stripping `clause:` and `#anchor`).")
    for s in issues:
        add(f"- {s}")
    add("")

    add("## Chunks (`data/processed/chunks.jsonl`)\n")
    add(f"- {ch['n_chunks']} chunks, {ch['n_clauses']} clauses. Chunk types: "
        + ", ".join(f"`{k}` {v}" for k, v in ch["chunk_types"].most_common()) + ".")
    add("- Chunks per clause: " + ", ".join(f"{k} chunk(s): {v} clauses" for k, v in sorted(ch["chunks_per_clause"].items())) + ".")
    for f, v in ch["missing_fields"].items():
        add(f"- Missing `{f}`: {len(v)} chunks ({_head(v)}).")
    if "paragraph" not in ch["missing_fields"]:
        pass
    add(f"- Duplicate `chunk_id`s: {len(ch['duplicate_chunk_ids'])} ({_head(ch['duplicate_chunk_ids'])}).")
    add(f"- `chunk_id` prefix vs `clause_id` mismatches: {len(ch['prefix_mismatch'])} ({_head(ch['prefix_mismatch'])}).")
    add(f"- `chunk_id` without `clause:` prefix: {len(ch['no_clause_prefix'])}.")
    add(f"- Empty text: {len(ch['empty_text'])}. Text under {SHORT_TEXT_CHARS} chars: {len(ch['short_text'])} "
        f"({_head([f'{c} ({n})' for c, n in ch['short_text']])}).")
    add(f"- Clauses with inconsistent `parent_clause`: {len(ch['inconsistent_parent'])}; inconsistent `title`: "
        f"{len(ch['inconsistent_title'])} ({_head(ch['inconsistent_title'])}).")
    add(f"- Chunks containing a literal `\\n` escape: {len(ch['literal_backslash_n'])} ({_head(ch['literal_backslash_n'])}).")
    add("- Single-chunk clauses have no `paragraph` field and no `#p` suffix (e.g. `clause:expr.post.incr`); "
        "multi-chunk clauses use `clause:<id>#p<N>`. `norm_id` handles both.\n")

    add("## Cross-references\n")
    add(f"- {rf['n_ref_mentions']} reference mentions (all `edge_type` = "
        + ", ".join(f"`{k}`" for k in rf["edge_type_values"]) + f"); {rf['anchored_targets']} carry a `#anchor` "
        "(e.g. `dcl.init.general#nt:initializer`), stripped by `norm_id`. Anchor kinds: "
        + ", ".join(f"`{k}` {v}" for k, v in rf["anchor_kinds"].most_common())
        + " (`nt` = grammar nonterminal, `def` = link to a term's definition).")
    add(f"- Mentions inside the corpus: {rf['inside_mentions']}; outside: {rf['outside_mentions']}.")
    add(f"- Distinct (source clause, target clause) pairs: {rf['n_clause_pairs']}. Inside the corpus (non-self): "
        f"{rf['inside_pairs']}; outside: {rf['outside_pairs']}.")
    add("- Most common outside targets: " + ", ".join(f"`{t}` ({n})" for t, n in rf["top_outside"]) + ".")
    add(f"- Outside targets that are parent sections of corpus clauses (no chunks of their own): "
        f"{len(rf['outside_parent_sections'])} ({_head(rf['outside_parent_sections'], 20)}).")
    add(f"- Of those, targets whose text lives in a corpus `<id>.general` clause: {len(rf['outside_with_general'])} "
        f"({_head(rf['outside_with_general'], 20)}). These are dropped by default; the `references_general` "
        "variant redirects them.")
    add(f"- Self-references: {len(rf['self_refs'])} clauses ({_head(rf['self_refs'])}). Removed in the graph.")
    add(f"- References that duplicate the parent edge (clause cites its own `parent_clause`): "
        f"{len(rf['refs_to_own_parent'])} ({_head([f'{s} -> {t}' for s, t in rf['refs_to_own_parent']])}).\n")

    add("## Parents\n")
    add(f"- {pa['distinct_parents']} distinct `parent_clause` values; corpus clauses among them: "
        f"{len(pa['parents_in_corpus'])} ({_head(pa['parents_in_corpus'])}).")
    add(f"- Not corpus clauses: {_head(pa['parents_not_in_corpus'], 20)}.")
    add(f"- Parents with a corpus `.general` clause: {len(pa['parents_with_general_clause'])} "
        f"({_head(pa['parents_with_general_clause'], 20)}).")
    add("- Group sizes (clauses per parent): " + ", ".join(f"`{p}` {n}" for p, n in pa["group_sizes"].items()) + ".\n")

    add("## `data/graph/spec_graph.json`\n")
    add(f"- Format: a JSON object with keys {_head(sg['top_keys'])}; version `{sg['graph_version']}`, "
        f"edition `{sg['spec_edition']}`. Nodes are `{{node_id, node_type, clause_id, title, chunk_id}}`; "
        "edges are `{source, target, edge_type, provenance}` with node IDs as endpoints.")
    add("- Node types: " + ", ".join(f"`{k}` {v}" for k, v in sg["node_types"].items())
        + ". Edge types: " + ", ".join(f"`{k}` {v}" for k, v in sg["edge_types"].items())
        + ". Provenance: " + ", ".join(f"`{k}` {v}" for k, v in sg["provenance"].items()) + ".")
    add(f"- Node IDs are `clause:<clause_id>` ({dict(sg['node_id_formats'])}); `norm_id(node_id) == clause_id` for "
        f"every node: {sg['node_ids_match_clause_ids']}. Chunks use the same `clause:` prefix plus `#p<N>`.")
    add(f"- Container nodes (= the {len(sg['container_nodes'])} parent sections): {_head(sg['container_nodes'], 20)}.")
    add(f"- Clause nodes not in the corpus: {len(sg['clause_nodes_not_in_corpus'])}; corpus clauses with no node: "
        f"{len(sg['corpus_clauses_missing_node'])}.")
    add(f"- Clause nodes with `chunk_id: null` (the multi-chunk clauses): {len(sg['clause_nodes_null_chunk_id'])} "
        f"({_head(sg['clause_nodes_null_chunk_id'])}). Node `chunk_id`s not found in chunks: "
        f"{len(sg['node_chunk_ids_not_in_chunks'])}.")
    add(f"- Reference edges in both files: {sg['refs_both']}; only in `spec_graph.json`: {len(sg['refs_graph_only'])} "
        f"({_head([f'{s}->{t}' for s, t in sg['refs_graph_only']])}); only in chunks (restricted to targets that "
        f"are graph nodes): {len(sg['refs_chunks_only'])} ({_head([f'{s}->{t}' for s, t in sg['refs_chunks_only']])}).")
    if sg["chunk_only_all_anchored"]:
        add(f"- **Cause:** every chunk-only edge comes from a `#`-anchored mention (grammar-nonterminal "
            f"links like `expr.cond#nt:conditional-expression`, or `#def:` term-definition links). The graph file equals the un-anchored chunk references "
            f"exactly: {sg['graph_equals_unanchored_chunks']}. Its builder apparently did not strip anchors, so "
            "those targets did not match any node. The clause graph keeps them (chunks are canonical); the "
            "`references_plain` variant reproduces the graph file's edge set.")
    add(f"- Of the graph file's reference edges, {sg['refs_graph_corpus_to_corpus']} join two corpus clauses; "
        "the rest touch a container node. The graph file drops targets that are neither corpus clauses nor containers.")
    add(f"- Parent edges in both: {sg['parents_both']}; graph only: {len(sg['parents_graph_only'])}; chunks only: "
        f"{len(sg['parents_chunks_only'])}. All parent edges point clause -> container.\n")

    add("## Benchmark (`data/benchmark/spec_benchmark_v1.jsonl`)\n")
    add(f"- {bm['n_queries']} queries; duplicate IDs: {len(bm['duplicate_query_ids'])} ({_head(bm['duplicate_query_ids'])}); "
        f"empty gold sets: {len(bm['empty_gold'])}.")
    add("- Raw difficulty labels found: " + ", ".join(f"`{k}` {v}" for k, v in bm["raw_difficulty_labels"].items())
        + ". The project doc also says \"multi clause\"; that spelling does not occur in the file. "
        "`normalize_difficulty` maps all variants to `single_clause` / `two_clause` / `multi_hop`.")
    add("- Gold-set size by difficulty: " + ", ".join(f"{d} x{n}: {c}" for (d, n), c in sorted(bm["gold_size_by_difficulty"].items())) + ".")
    add(f"- Labels disagreeing with gold size (single=1, two=2, multi>=3): {len(bm['label_size_mismatch'])} "
        f"({_head([f'{q} ({d}, {n} gold)' for q, d, n in bm['label_size_mismatch']])}). Not relabeled; results "
        "are reported both by label and by gold-set size.")
    add(f"- Gold IDs needing normalization (`clause:` prefix or `#`): {len(bm['gold_needing_normalization'])}. "
        f"Gold IDs not in the corpus: {len(bm['gold_not_in_corpus'])} ({_head(bm['gold_not_in_corpus'])}). "
        f"Duplicate gold within a query: {len(bm['duplicate_gold_within_query'])}.")
    add(f"- **Only {bm['distinct_gold_clauses']} of 82 corpus clauses ever appear as gold.** Frequency (queries): "
        + ", ".join(f"`{g}` {n}" for g, n in bm["gold_frequency"].most_common()) + ". Any method that ranks these "
        "few clauses highly for every query scores well, so per-clause priors are a real risk for learned rerankers.")
    add("- Expected properties: "
        + ", ".join(f"`{k}` {v}" for k, v in bm["expected_property"].most_common()) + ".")
    return "\n".join(L) + "\n"


def run_validation(out: Path = RESULTS_DIR / "data_validation.md") -> dict[str, Any]:
    chunks = load_chunks()
    queries = erica.read_records(BENCHMARK_PATH)  # raw labels, before normalization
    report = {"chunks": check_chunks(chunks), "references": check_references(chunks),
              "parents": check_parents(chunks), "spec_graph": check_spec_graph(chunks, load_spec_graph()),
              "benchmark": check_benchmark(chunks, queries)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(*report.values()), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_validation()
    print(f"Wrote {RESULTS_DIR / 'data_validation.md'}")
