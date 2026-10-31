"""One command for the milestone graph results.

    python graph/specgraph_graph/run_milestone.py

Needs numpy only. Rebuilds the validation report and clause graph, loads the
cached full clause scores (artifacts/scores/question_scores.json; regenerate
with clause_scores.py) and Erica's saved question-only rankings, and writes:

    results/milestone/graph_expansion_results.md
    results/milestone/per_query.csv
    results/milestone/run_config.json

The Dense + GraphExp defaults below were fixed before any result was seen.
The λ × hops grid is exploratory; nothing is selected from it.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from clause_graph import DEFAULT_EDGE_TYPES, adjacency, build_clause_graph, node_ids, save_clause_graph
from clause_scores import SCORES_PATH, load_scores
from common import (BASELINE_DIR, BENCHMARK_PATH, CHUNKS_PATH, CLAUSE_GRAPH_PATH, DIFFICULTIES,
                    PROJECT_ROOT, RESULTS_DIR, load_benchmark, load_chunks, write_json)
from evaluate import KS, boot_ci, evaluate_method, mean_by_group, paired, reachability
from graph_expand import graph_expand, normalized_adjacency, rank, top_k_indices
from validate_data import run_validation

DEFAULTS = {"seed_k": 5, "hops": 1, "lam": 0.5, "edge_types": DEFAULT_EDGE_TYPES}
GRID_LAMBDAS = (0.25, 0.5, 1.0)
GRID_HOPS = (1, 2)
RRF_K, CANDIDATE_K = 60, 50

# Erica's saved question-only rankings (her reported reference input mode).
ERICA_RUNS = {
    "BM25": ("results_question_only", "bm25"),
    "Dense": ("results_question_only", "dense"),
    "Hybrid RRF 1:1": ("results_question_only", "hybrid_rrf"),
    "Hybrid RRF 2:1": ("results_ablation_question", "hybrid_rrf_d2"),
}


def load_erica_rankings() -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for name, (run, key) in ERICA_RUNS.items():
        rows = [json.loads(line) for line in (BASELINE_DIR / run / "ranked_results.jsonl").open(encoding="utf-8")]
        out[name] = {r["query_id"]: r["results"][key] for r in rows}
    return out


def rrf_scores(bm25: np.ndarray, dense: np.ndarray, ids: list[str], dense_weight: float) -> np.ndarray:
    """Erica's clause RRF as a full score vector: each branch truncated to its top-50 clauses."""
    out = np.zeros(len(ids))
    for scores, w in ((bm25, 1.0), (dense, dense_weight)):
        for r, i in enumerate(top_k_indices(scores, CANDIDATE_K), start=1):
            out[i] += w / (RRF_K + r)
    return out


def expand_all(base: np.ndarray, graph: dict[str, Any], qids: list[str], **kw: Any) -> dict[str, list[str]]:
    ids = node_ids(graph)
    a_norm = normalized_adjacency(adjacency(graph, kw.pop("edge_types", DEFAULT_EDGE_TYPES)))
    # Â is computed once per configuration and passed to graph_expand directly.
    return {qid: rank(graph_expand(base[i], a_norm, **kw), ids) for i, qid in enumerate(qids)}


def pct(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{100 * x:.1f}"


def ci(t: tuple[float, float, float]) -> str:
    return f"{pct(t[0])} [{pct(t[1])}, {pct(t[2])}]"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    run_validation()
    graph = build_clause_graph(load_chunks())
    save_clause_graph(graph)
    ids = node_ids(graph)
    queries = load_benchmark()
    qids = [q["query_id"] for q in queries]
    scores = load_scores()
    if scores["clause_ids"] != ids or scores["query_ids"] != qids:
        raise ValueError("Cached scores do not match the corpus/benchmark order; rerun clause_scores.py")
    dense, bm25 = scores["dense"], scores["bm25"]

    rankings = load_erica_rankings()
    # The cached scores must reproduce Erica's saved Dense/BM25 rankings exactly.
    for name, mat in (("Dense", dense), ("BM25", bm25)):
        bad = [q for i, q in enumerate(qids) if rank(mat[i], ids)[:CANDIDATE_K] != rankings[name][q]]
        if bad:
            raise ValueError(f"Cached {name} scores disagree with Erica's saved ranking for {bad[:5]}")
    hyb2 = np.stack([rrf_scores(bm25[i], dense[i], ids, 2.0) for i in range(len(qids))])
    bad = [q for i, q in enumerate(qids) if rank(hyb2[i], ids)[:CANDIDATE_K] != rankings["Hybrid RRF 2:1"][q]]
    if bad:
        raise ValueError(f"Reconstructed 2:1 RRF disagrees with Erica's saved ranking for {bad[:5]}")

    main_name = "Dense + GraphExp"
    rankings[main_name] = expand_all(dense, graph, qids, **DEFAULTS)

    variants: dict[str, dict[str, list[str]]] = {f"{main_name} (default: refs+parent+sibling, 1 hop)": rankings[main_name]}
    variant_specs = [
        ("hops = 2", {"hops": 2}),
        ("references only", {"edge_types": ("references",)}),
        ("sibling only", {"edge_types": ("sibling",)}),
        ("parent only (no edges on this data = Dense)", {"edge_types": ("parent",)}),
        ("references_plain only (= spec_graph.json refs)", {"edge_types": ("references_plain",)}),
        ("refs + references_general + sibling", {"edge_types": ("references", "references_general", "sibling")}),
    ]
    for label, override in variant_specs:
        variants[f"{main_name}, {label}"] = expand_all(dense, graph, qids, **{**DEFAULTS, **override})
    variants["Hybrid 2:1 + GraphExp (hybrid seeds)"] = expand_all(hyb2, graph, qids, **DEFAULTS)

    grid = {(lam, h): expand_all(dense, graph, qids, **{**DEFAULTS, "lam": lam, "hops": h})
            for lam in GRID_LAMBDAS for h in GRID_HOPS}

    evals = {m: evaluate_method(r, queries) for m, r in {**rankings, **variants}.items()}
    grid_evals = {k: evaluate_method(r, queries) for k, r in grid.items()}
    by_diff = lambda q: q["difficulty_norm"]  # noqa: E731
    by_size = lambda q: len(q["gold"])  # noqa: E731

    md: list[str] = []
    add = md.append
    add("# Graph-expansion results (milestone)\n")
    add(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `python graph/specgraph_graph/run_milestone.py`.\n")
    add("**Setup.** 49 benchmark queries, 82 clauses, question-only input (Erica's reference mode). "
        "BM25 / Dense / Hybrid rows are Erica's saved rankings (`baseline/specgraph_baselines/results_question_only`, "
        "`results_ablation_question`); all rows go through the same metric code (`retrieval_baselines.metrics_at_k`). "
        "Dense + GraphExp reranks Dense's full 82-clause scores (clause = max over chunks; these reproduce her saved "
        "Dense ranking exactly) with seed_k=5, hops=1, λ=0.5, edges = references + parent + sibling, fixed in advance. "
        "\"Hybrid\" here is Erica's BM25+Dense reciprocal-rank fusion. Graph expansion is a separate method, "
        "always called **Dense + GraphExp**. Coverage = complete coverage (all gold clauses in top-k). "
        "All numbers are percentages.\n")
    add("> **Headline metric:** the project doc says \"coverage@5\" in one place and \"every gold clause in the "
        "top 10\" in another, so both are reported. Coverage@10 is the headline.\n")

    add("## Main table\n")
    add("| Method | Cov@1 | Cov@3 | Cov@5 | **Cov@10** | R@1 | R@3 | R@5 | R@10 | P@1 | P@3 | P@5 | P@10 | nDCG@10 |")
    add("|---|" + "---:|" * 13)
    for m in rankings:
        a = mean_by_group(evals[m], queries)["all"]
        cells = ([pct(a[f"complete_coverage@{k}"]) for k in KS] + [pct(a[f"recall@{k}"]) for k in KS]
                 + [pct(a[f"precision@{k}"]) for k in KS] + [f"{a['ndcg@10']:.3f}"])
        cells[3] = f"**{cells[3]}**"
        add(f"| {m} | " + " | ".join(cells) + " |")
    add("")

    add("## By difficulty label\n")
    add("| Method | " + " | ".join(f"{d} Cov@5 | {d} Cov@10 | {d} R@5 | {d} R@10" for d in DIFFICULTIES) + " |")
    add("|---|" + "---:|" * 12)
    for m in rankings:
        g = mean_by_group(evals[m], queries, by_diff)
        add(f"| {m} | " + " | ".join(" | ".join(pct(g[d][x]) for x in ("complete_coverage@5", "complete_coverage@10",
                                                                     "recall@5", "recall@10")) for d in DIFFICULTIES) + " |")
    n_diff = {d: sum(q["difficulty_norm"] == d for q in queries) for d in DIFFICULTIES}
    add(f"\nn = {n_diff}. Eight `multi_hop` queries have only 2 gold clauses (see data_validation.md), so the "
        "table below groups by gold-set size as well.\n")
    sizes = sorted({len(q["gold"]) for q in queries})
    add("| Method | " + " | ".join(f"{s} gold Cov@10 | {s} gold R@10" for s in sizes) + " |")
    add("|---|" + "---:|" * (2 * len(sizes)))
    for m in rankings:
        g = mean_by_group(evals[m], queries, by_size)
        add(f"| {m} | " + " | ".join(f"{pct(g[s]['complete_coverage@10'])} | {pct(g[s]['recall@10'])}" for s in sizes) + " |")
    add(f"\nn = { {s: sum(len(q['gold']) == s for q in queries) for s in sizes} }.\n")

    add("## Uncertainty (95% bootstrap CI, 2000 query resamples, seed 0)\n")
    add("| Method | Cov@10 | R@5 | multi_hop Cov@10 |")
    add("|---|---:|---:|---:|")
    mh = [q for q in queries if q["difficulty_norm"] == "multi_hop"]
    for m in rankings:
        e = evals[m]
        add(f"| {m} | {ci(boot_ci([e[q]['complete_coverage@10'] for q in qids]))} | "
            f"{ci(boot_ci([e[q]['recall@5'] for q in qids]))} | "
            f"{ci(boot_ci([e[q['query_id']]['complete_coverage@10'] for q in mh]))} |")
    add("")
    add("**Paired: Dense + GraphExp vs Dense**, per query (wins / losses / ties, and the bootstrap CI of the mean "
        "difference in points):\n")
    add("| Subset | Metric | Wins | Losses | Ties | Mean diff [95% CI] |")
    add("|---|---|---:|---:|---:|---:|")
    paired_out = {}
    for subset_name, subset in (("all", queries), ("multi_hop", mh)):
        for metric in ("complete_coverage@10", "complete_coverage@5", "recall@5", "recall@10"):
            p = paired([evals[main_name][q["query_id"]][metric] for q in subset],
                       [evals["Dense"][q["query_id"]][metric] for q in subset])
            paired_out[f"{subset_name}/{metric}"] = p
            add(f"| {subset_name} (n={len(subset)}) | {metric} | {p['wins']} | {p['losses']} | {p['ties']} | "
                f"{100 * p['mean_diff']:+.1f} [{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}] |")
    add("")

    add("## Variants (exploratory)\n")
    add("| Variant | Cov@5 | Cov@10 | R@5 | R@10 | multi_hop Cov@10 | multi_hop R@10 | nDCG@10 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in ["Dense", *variants]:
        a = mean_by_group(evals[m], queries)["all"]
        g = mean_by_group(evals[m], queries, by_diff)["multi_hop"]
        add(f"| {m} | {pct(a['complete_coverage@5'])} | {pct(a['complete_coverage@10'])} | {pct(a['recall@5'])} | "
            f"{pct(a['recall@10'])} | {pct(g['complete_coverage@10'])} | {pct(g['recall@10'])} | {a['ndcg@10']:.3f} |")
    add("\n`references_plain` is the reference edge set of `data/graph/spec_graph.json` (it drops the 97 "
        "`#`-anchored references). `references_general` redirects references to chunk-less parent sections "
        "(e.g. `over.oper`) to that section's `.general` clause.\n")

    add("## Sensitivity grid (exploratory, Dense + GraphExp, default edges, seed_k=5)\n")
    add("Not used to choose a setting: choosing from this grid would be tuning on the test set. "
        "Model selection needs a dev/test split.\n")
    add("| λ | hops | Cov@5 | Cov@10 | R@5 | R@10 | multi_hop Cov@10 |")
    add("|---:|---:|---:|---:|---:|---:|---:|")
    for (lam, h), e in grid_evals.items():
        a = mean_by_group(e, queries)["all"]
        g = mean_by_group(e, queries, by_diff)["multi_hop"]
        add(f"| {lam} | {h} | {pct(a['complete_coverage@5'])} | {pct(a['complete_coverage@10'])} | "
            f"{pct(a['recall@5'])} | {pct(a['recall@10'])} | {pct(g['complete_coverage@10'])} |")
    add("")

    add("## Reachability from the Dense top-5 (1 hop)\n")
    add("*Gold in reach*: share of a query's gold clauses in seeds ∪ N(seeds); *chance*: share of all 82 clauses "
        "in seeds ∪ N(seeds). *Missed gold found*: of the gold clauses Dense did not put in its top 5, the share that "
        "are graph neighbors of the seeds; *chance*: share of all non-seed clauses that are neighbors. "
        "Macro over queries (missed-gold columns only over queries with a missed gold clause).\n")
    add("| Edges | Difficulty | n | Gold in reach | Chance | Missed gold found | Chance (non-seed) | n w/ missed |")
    add("|---|---|---:|---:|---:|---:|---:|---:|")
    reach_out: dict[str, Any] = {}
    for label, types in (("refs+parent+sibling", DEFAULT_EDGE_TYPES), ("references", ("references",)),
                         ("sibling", ("sibling",))):
        adj = adjacency(graph, types)
        rows = {q["query_id"]: reachability(rankings["Dense"][q["query_id"]][:5], q["gold"], adj, ids) for q in queries}
        for d in ("all", *DIFFICULTIES):
            sub = [q for q in queries if d == "all" or q["difficulty_norm"] == d]
            r = [rows[q["query_id"]] for q in sub]
            withm = [x for x in r if x["n_missed"]]
            vals = (np.mean([x["gold_in_reach"] for x in r]), np.mean([x["chance_reach"] for x in r]),
                    np.mean([x["missed_found"] for x in withm]) if withm else float("nan"),
                    np.mean([x["chance_missed"] for x in withm]) if withm else float("nan"))
            reach_out[f"{label}/{d}"] = vals
            add(f"| {label} | {d} | {len(sub)} | " + " | ".join(pct(v) for v in vals) + f" | {len(withm)} |")
    add("")

    with (RESULTS_DIR / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        method_cols = list(rankings) + list(variants)[1:]
        w.writerow(["query_id", "difficulty", "n_gold", "gold"] + [f"cov@10 {m}" for m in method_cols]
                   + ["dense_top10", "graphexp_top10"])
        for q in queries:
            qid = q["query_id"]
            w.writerow([qid, q["difficulty_norm"], len(q["gold"]), ";".join(q["gold"])]
                       + [int(evals[m][qid]["complete_coverage@10"]) for m in method_cols]
                       + [";".join(rankings["Dense"][qid][:10]), ";".join(rankings[main_name][qid][:10])])

    (RESULTS_DIR / "graph_expansion_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    import numpy
    write_json(RESULTS_DIR / "run_config.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python graph/specgraph_graph/run_milestone.py",
        "graph_expand_defaults": {**DEFAULTS, "edge_types": list(DEFAULT_EDGE_TYPES)},
        "grid": {"lambda": GRID_LAMBDAS, "hops": GRID_HOPS},
        "variants": {label: override for label, override in variant_specs},
        "ks": KS, "bootstrap": {"samples": 2000, "seed": 0},
        "query_mode": scores["query_mode"], "embedding_model": scores["embedding_model"],
        "clause_score": scores["clause_score"], "erica_runs": ERICA_RUNS,
        "rrf": {"k": RRF_K, "candidate_k": CANDIDATE_K, "hybrid_seed_dense_weight": 2.0},
        "packages": {"numpy": numpy.__version__, **{f"scores:{k}": v for k, v in scores["packages"].items()}},
        "input_sha256": {str(p.relative_to(PROJECT_ROOT)): sha(p) for p in
                         (CHUNKS_PATH, BENCHMARK_PATH, CLAUSE_GRAPH_PATH, SCORES_PATH)},
        "paired_graphexp_vs_dense": paired_out,
        "reachability": reach_out,
    })
    print("\n".join(md))


if __name__ == "__main__":
    main()
