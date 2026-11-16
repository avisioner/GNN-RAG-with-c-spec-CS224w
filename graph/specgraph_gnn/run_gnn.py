"""One command for the GNN milestone results (old 82-clause corpus, 49 real benchmark queries).

    baseline/.venv/bin/python graph/specgraph_gnn/run_gnn.py [--workers 6]

Real data only: no synthetic or generated queries.

Protocol: 5-fold cross-validation over the 49 benchmark queries, stratified by
difficulty. For test fold f, fold (f+1) mod 5 is the validation fold (early
stopping only) and the other 3 folds are training. Every query is scored once,
by a model that never saw it. Hyperparameters are fixed in advance (brief §7);
nothing is tuned on results.

Steps: score the benchmark questions with Erica's BM25/dense functions (must
match artifacts/scores/question_scores.json) -> features -> train every
model x seed x fold (cached in results/gnn_milestone/cache/) -> evaluate with
Erica's metrics_at_k -> write results/gnn_milestone/RESULTS.md, per_query.csv,
run_config.json.

Clause-popularity prior: a non-learned check. For each test fold it ranks
clauses by how often they are gold in the other 4 folds (ties by dense score).
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from gnn_common import CORPUS_V1_COMMIT, CORPUS_V1_PATH, OUT_DIR, PROJECT_ROOT, graph_common, load_benchmark, sha256, \
    write_json
from features import RELATION_SETS, build_features, build_relations, degree_feature_for, load_clause_graph, node_ids
from scoring import BGE_PREFIX, LEXICAL_MODE, MODEL_NAME, benchmark_scores
from train import TrainConfig, gold_mask, train_and_predict

import clause_scores  # noqa: E402
from evaluate import KS, boot_ci, evaluate_method  # noqa: E402
from graph_expand import rank  # noqa: E402
from run_milestone import DEFAULTS as GRAPHEXP_DEFAULTS, expand_all, load_erica_rankings  # noqa: E402

SEEDS = (0, 1, 2)
N_FOLDS = 5
FOLD_SEED = 0
DIFFS = graph_common.DIFFICULTIES

# Display name -> (model kind, relation set). The MLP gets the refs+sibling degree feature
# (identical features to R-GCN) but never sees an edge.
MODELS: dict[str, tuple[str, str]] = {
    "MLP (no graph)": ("mlp", "refs+sibling"),
    "R-GCN": ("rgcn", "refs+sibling"),
    "R-GCN (no sibling)": ("rgcn", "refs only"),
    "GraphSAGE": ("sage", "refs+sibling"),
    "GraphSAGE (no sibling)": ("sage", "refs only"),
    "GAT": ("gat", "refs+sibling"),
    "GAT (no sibling)": ("gat", "refs only"),
}
SIBLING_PAIRS = (("R-GCN", "R-GCN (no sibling)"), ("GraphSAGE", "GraphSAGE (no sibling)"), ("GAT", "GAT (no sibling)"))
MAIN = "R-GCN"
PRIOR = "Clause-popularity prior"
BASELINES = ("BM25", "Dense", "Hybrid RRF 1:1", "Hybrid RRF 2:1", "Dense + GraphExp")


def stratified_folds(queries: list[dict[str, Any]], k: int = N_FOLDS, seed: int = FOLD_SEED) -> dict[str, int]:
    rng = random.Random(seed)
    fold: dict[str, int] = {}
    start = 0
    for d in DIFFS:
        qids = sorted(q["query_id"] for q in queries if q["difficulty_norm"] == d)
        rng.shuffle(qids)
        for i, qid in enumerate(qids):
            fold[qid] = (start + i) % k
        start += len(qids)          # continue the round-robin so fold sizes stay balanced
    return fold


def val_fold(test_fold: int) -> int:
    return (test_fold + 1) % N_FOLDS


def _job(args: tuple) -> dict[str, Any]:
    return train_and_predict(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    opts = ap.parse_args()

    graph = load_clause_graph()
    ids = node_ids(graph)
    n = len(ids)
    index = {c: i for i, c in enumerate(ids)}
    queries = load_benchmark()
    qids = [q["query_id"] for q in queries]
    qby = {q["query_id"]: q for q in queries}

    bench_scores = benchmark_scores()          # raises unless it matches question_scores.json
    g_bench = gold_mask([[index[g] for g in q["gold"]] for q in queries], n)
    feats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]] = {}
    for rs_name, rs in RELATION_SETS.items():
        ei, et, rel_names = build_relations(graph, rs)
        feats[rs_name] = (build_features(bench_scores, degree_feature_for(graph, rs)), ei, et, rel_names)

    folds = stratified_folds(queries)
    pos = lambda fs: [i for i, q in enumerate(qids) if folds[q] in fs]  # noqa: E731
    jobs, keys = [], []
    for name in opts.models:
        kind, rs_name = MODELS[name]
        x, ei, et, rel_names = feats[rs_name]
        cfg = TrainConfig(kind=kind, relation_set=rs_name)
        for seed in SEEDS:
            for f in range(N_FOLDS):
                v = val_fold(f)
                tr, va, te = pos(set(range(N_FOLDS)) - {f, v}), pos({v}), pos({f})
                jobs.append((cfg, seed, x[tr], g_bench[tr], x[va], g_bench[va], x[te], ei, et, len(rel_names),
                             f"{name}|{seed}|{f}"))
                keys.append((name, seed, f))

    print(f"{len(jobs)} training runs ({opts.workers} workers)")
    with ProcessPoolExecutor(max_workers=opts.workers) as pool:
        recs = list(pool.map(_job, jobs))
    print(f"trained {sum(not r['cached'] for r in recs)}, cached {sum(r['cached'] for r in recs)}")

    rankings: dict[str, dict[Any, dict[str, list[str]]]] = defaultdict(lambda: defaultdict(dict))
    val_r5: dict[str, list[float]] = defaultdict(list)
    epochs: dict[str, list[int]] = defaultdict(list)
    for (name, seed, f), rec in zip(keys, recs):
        val_r5[name].append(rec["val_recall@5"])
        epochs[name].append(rec["best_epoch"])
        for i, qid in enumerate([q for q in qids if folds[q] == f]):
            rankings[name][seed][qid] = rank(rec["eval_scores"][i], ids)

    erica_rank = load_erica_rankings()
    dense = clause_scores.load_scores()["dense"]
    erica_rank["Dense + GraphExp"] = expand_all(dense, graph, qids, **dict(GRAPHEXP_DEFAULTS))

    # Clause-popularity prior: gold frequency in the 4 non-test folds, ties by dense score, then clause ID.
    prior: dict[str, list[str]] = {}
    set_seen: dict[str, bool] = {}
    clauses_seen: dict[str, bool] = {}
    popular: dict[str, set[str]] = {}
    for f in range(N_FOLDS):
        others = [o for o in qids if folds[o] != f]
        counts = Counter(g for o in others for g in qby[o]["gold"])
        sets = {frozenset(qby[o]["gold"]) for o in others}
        for i, qid in enumerate(qids):
            if folds[qid] == f:
                prior[qid] = sorted(ids, key=lambda c: (-counts[c], -dense[i][index[c]], index[c]))
                clauses_seen[qid] = all(counts[g] > 0 for g in qby[qid]["gold"])
                set_seen[qid] = frozenset(qby[qid]["gold"]) in sets
                popular[qid] = set(counts)

    pq: dict[str, dict[Any, dict[str, dict[str, float]]]] = {}
    for name in BASELINES:
        pq[name] = {None: evaluate_method(erica_rank[name], queries)}
    pq[PRIOR] = {None: evaluate_method(prior, queries)}
    for name, by_seed in rankings.items():
        pq[name] = {s: evaluate_method(r, queries) for s, r in by_seed.items()}

    # Baselines must match the Oct 30 milestone per-query coverage@10 exactly.
    with (PROJECT_ROOT / "results" / "milestone" / "per_query.csv").open(encoding="utf-8") as fh:
        old = {r["query_id"]: r for r in csv.DictReader(fh)}
    for name in BASELINES:
        bad = [q for q in qids if float(old[q][f"cov@10 {name}"]) != pq[name][None][q]["complete_coverage@10"]]
        if bad:
            raise ValueError(f"{name} disagrees with results/milestone/per_query.csv on {bad[:5]}")

    all_rank: dict[str, dict[Any, dict[str, list[str]]]] = {b: {None: erica_rank[b]} for b in BASELINES}
    all_rank[PRIOR] = {None: prior}
    all_rank.update(rankings)
    pop_share = {k: float(np.mean([np.mean([len(set(r[q][:10]) & popular[q]) / 10 for q in qids])
                                   for r in by_seed.values()])) for k, by_seed in all_rank.items()}

    write_outputs(pq, queries, folds, clauses_seen, set_seen, pop_share, val_r5, epochs,
                  rel_info={k: v[3] for k, v in feats.items()}, n_edges={k: int(v[1].shape[1]) for k, v in feats.items()},
                  models=opts.models, n_runs=len(jobs))


# --------------------------------------------------------------------------- reporting

def seed_mean_per_query(by_seed: dict[Any, dict[str, dict[str, float]]], metric: str, qids: list[str]) -> np.ndarray:
    return np.mean([[by_seed[s][q][metric] for q in qids] for s in by_seed], axis=0)


def agg(by_seed: dict[Any, dict[str, dict[str, float]]], metric: str, qids: list[str]) -> tuple[float, float | None]:
    """(mean, std) over seeds of the macro average over `qids`; std None for single-run methods."""
    vals = [np.mean([by_seed[s][q][metric] for q in qids]) for s in by_seed]
    return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else None


def fmt(m: float, s: float | None) -> str:
    return f"{100 * m:.1f}" if s is None else f"{100 * m:.1f} ± {100 * s:.1f}"


def write_outputs(pq, queries, folds, clauses_seen, set_seen, pop_share, val_r5, epochs, rel_info, n_edges,
                  models, n_runs) -> None:
    qids = [q["query_id"] for q in queries]
    by_diff = {d: [q["query_id"] for q in queries if q["difficulty_norm"] == d] for d in DIFFS}
    rows = list(BASELINES) + [PRIOR] + list(models)
    fold_sizes = Counter(folds.values())
    md: list[str] = []
    md += ["# GNN reranker: milestone results", "",
           f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `graph/specgraph_gnn/run_gnn.py`. "
           "Old 82-clause corpus (`artifacts/corpus_v1/chunks.jsonl`, = commit 7a29ead), 49 benchmark queries, "
           "question-only input. Metrics from Erica's `metrics_at_k`. Rankings over all 82 clauses, ties by clause ID.", "",
           "- **Real data only.** GNN/MLP rows: 5-fold cross-validation over the 49 benchmark queries, stratified by "
           f"difficulty (fold sizes {sorted(fold_sizes.values())}). Per test fold: 3 folds train, 1 fold for early "
           "stopping, 1 fold test. Every query is scored once by a model that never saw it.",
           f"- Model rows: mean ± std over seeds {list(SEEDS)}. Baselines are deterministic and use no training.",
           f"- **{PRIOR}** is not a model: it ranks clauses by how often they are gold in the other 4 folds, "
           "ignoring the question (dense score only breaks ties). It measures how far one can get by memorizing "
           "which clauses are popular.",
           "- Headline metric: **coverage@10** (all gold clauses in the top 10). With 17 two-clause and 17 multi-hop "
           "queries, one query moves a per-difficulty number by ~5.9 points.",
           "- Single-clause queries are at 100% coverage@5 for every text baseline, so all differences come from "
           "two-clause and multi-hop queries.", ""]

    md += ["## Main table", "",
           "| Method | Cov@1 | Cov@3 | Cov@5 | **Cov@10** | R@1 | R@3 | R@5 | R@10 | P@5 | nDCG@10 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in rows:
        bs = pq[k]
        cells = [fmt(*agg(bs, f"complete_coverage@{kk}", qids)) for kk in KS]
        cells[3] = f"**{cells[3]}**"
        cells += [fmt(*agg(bs, f"recall@{kk}", qids)) for kk in KS]
        cells += [fmt(*agg(bs, "precision@5", qids)), f"{agg(bs, 'ndcg@10', qids)[0]:.3f}"]
        md.append(f"| {k} | " + " | ".join(cells) + " |")

    md += ["", "## By difficulty (Cov@5 / Cov@10 / R@5 / R@10)", "",
           "| Method | " + " | ".join(f"{d} {m}" for d in DIFFS for m in ("Cov@5", "Cov@10", "R@5", "R@10")) + " |",
           "|---|" + "---:|" * 12]
    for k in rows:
        cells = [fmt(*agg(pq[k], met, by_diff[d])) for d in DIFFS
                 for met in ("complete_coverage@5", "complete_coverage@10", "recall@5", "recall@10")]
        md.append(f"| {k} | " + " | ".join(cells) + " |")

    md += ["", "## Uncertainty (95% bootstrap CI over queries, 2000 resamples, seed 0; model rows use seed-averaged "
           "per-query values)", "", "| Method | Cov@10 | R@5 | multi_hop Cov@10 |", "|---|---:|---:|---:|"]
    for k in rows:
        cells = []
        for met, qs in (("complete_coverage@10", qids), ("recall@5", qids),
                        ("complete_coverage@10", by_diff["multi_hop"])):
            m, lo, hi = boot_ci(seed_mean_per_query(pq[k], met, qs))
            cells.append(f"{100 * m:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
        md.append(f"| {k} | " + " | ".join(cells) + " |")

    if MAIN in pq:
        md += ["", f"## Paired: {MAIN} vs others (per query; wins/losses/ties per seed, CI on the seed-averaged "
               "difference)", "", "| Comparison | Metric | Subset | W/L/T per seed | Mean diff [95% CI] |",
               "|---|---|---|---|---:|"]
        for other in ("Dense", "Dense + GraphExp", "Hybrid RRF 2:1", PRIOR, "MLP (no graph)", "R-GCN (no sibling)"):
            if other not in pq:
                continue
            single = None in pq[other]
            for met in ("complete_coverage@10", "recall@5"):
                for subset, qs in (("all", qids), ("multi_hop", by_diff["multi_hop"])):
                    b = seed_mean_per_query(pq[other], met, qs)
                    wlt = []
                    for s in sorted(pq[MAIN]):
                        a = np.array([pq[MAIN][s][q][met] for q in qs])
                        bb = b if single else np.array([pq[other][s][q][met] for q in qs])
                        wlt.append(f"{int((a > bb).sum())}/{int((a < bb).sum())}/{int((a == bb).sum())}")
                    m, lo, hi = boot_ci(seed_mean_per_query(pq[MAIN], met, qs) - b)
                    md.append(f"| {MAIN} vs {other} | {met.replace('complete_coverage', 'Cov')} | {subset} | "
                              f"{', '.join(wlt)} | {100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")

    md += ["", "## Sibling ablation by architecture", "",
           "Same model with vs without sibling edges (refs-only graph; the degree feature also drops siblings). "
           "Diff = with − without, seed-averaged per query, 95% bootstrap CI. W/L/T per seed pairs seed s with seed s.",
           "", "| Architecture | Metric | Subset | With sibling | Without sibling | W/L/T per seed | Diff [95% CI] |",
           "|---|---|---|---:|---:|---|---:|"]
    for with_s, without_s in SIBLING_PAIRS:
        if with_s not in pq or without_s not in pq:
            continue
        for met in ("complete_coverage@10", "recall@5"):
            for subset, qs in (("all", qids), ("multi_hop", by_diff["multi_hop"])):
                wlt = []
                for s in sorted(pq[with_s]):
                    a = np.array([pq[with_s][s][q][met] for q in qs])
                    b = np.array([pq[without_s][s][q][met] for q in qs])
                    wlt.append(f"{int((a > b).sum())}/{int((a < b).sum())}/{int((a == b).sum())}")
                m, lo, hi = boot_ci(seed_mean_per_query(pq[with_s], met, qs) - seed_mean_per_query(pq[without_s], met, qs))
                md.append(f"| {with_s} | {met.replace('complete_coverage', 'Cov')} | {subset} | "
                          f"{fmt(*agg(pq[with_s], met, qs))} | {fmt(*agg(pq[without_s], met, qs))} | {', '.join(wlt)} | "
                          f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")

    s_seen = [q for q in qids if set_seen[q]]
    s_unseen = [q for q in qids if not set_seen[q]]
    md += ["", "## Benchmark issue: clause popularity", "",
           f"The 49 queries use only {len({g for q in queries for g in q['gold']})} distinct gold clauses "
           f"(`expr.post.incr` is gold in {sum('expr.post.incr' in q['gold'] for q in queries)}). "
           f"{sum(clauses_seen.values())}/49 test queries have every gold clause already gold in another fold, so a "
           "split by unseen *clauses* is empty. The table splits instead by whether the query's exact gold **set** "
           f"occurs in another fold ({len(s_seen)} seen, {len(s_unseen)} unseen). *Popular@10* = mean share of the "
           "top 10 taken by clauses that are gold in another fold.", "",
           f"| Method | Cov@10 all | Cov@10 seen set (n={len(s_seen)}) | Cov@10 unseen set (n={len(s_unseen)}) | "
           "R@5 seen set | R@5 unseen set | Popular@10 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for k in ["Dense", "Dense + GraphExp", "Hybrid RRF 2:1", PRIOR] + list(models):
        cells = [fmt(*agg(pq[k], "complete_coverage@10", qids)), fmt(*agg(pq[k], "complete_coverage@10", s_seen)),
                 fmt(*agg(pq[k], "complete_coverage@10", s_unseen)), fmt(*agg(pq[k], "recall@5", s_seen)),
                 fmt(*agg(pq[k], "recall@5", s_unseen)), f"{100 * pop_share[k]:.0f}%"]
        md.append(f"| {k} | " + " | ".join(cells) + " |")

    md += ["", "## Early stopping", "",
           "Validation fold = the fold after the test fold. Max 100 epochs, patience 15. With about 29 training "
           "queries and batch size 64, one epoch is one gradient step.", "",
           "| Model | Val-fold R@5 at best epoch | Test R@5 | Best epoch (mean, range) |", "|---|---:|---:|---:|"]
    for m in models:
        v, e = val_r5[m], epochs[m]
        md.append(f"| {m} | {100 * np.mean(v):.1f} ± {100 * np.std(v):.1f} | {fmt(*agg(pq[m], 'recall@5', qids))} | "
                  f"{np.mean(e):.0f} ({min(e)}–{max(e)}) |")

    md += ["", "## Graph", ""]
    for rs, names in rel_info.items():
        md.append(f"- `{rs}`: relations {names}; {n_edges[rs]} directed edges.")
    md += ["- `parent` is in `default_edge_types` but has 0 edges on this corpus (no parent section is a "
           "corpus clause), so it contributes no relation.", ""]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    cols = ["query_id", "difficulty", "n_gold", "gold", "fold", "gold_clauses_seen_in_other_folds",
            "gold_set_seen_in_other_folds"]
    metric_cols = []
    for k in rows:
        for met in ("complete_coverage@10", "recall@5"):
            metric_cols.append((k, met))
            cols.append(f"{met.replace('complete_coverage', 'cov')} {k}")
    with (OUT_DIR / "per_query.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for q in queries:
            qid = q["query_id"]
            row = [qid, q["difficulty_norm"], len(q["gold"]), ";".join(q["gold"]), folds[qid],
                   int(clauses_seen[qid]), int(set_seen[qid])]
            row += [round(float(np.mean([pq[k][s][qid][met] for s in pq[k]])), 4) for k, met in metric_cols]
            w.writerow(row)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                                text=True).stdout.strip())
    write_json(OUT_DIR / "run_config.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "working_tree_dirty": dirty,
        "data": "real benchmark queries only; no synthetic queries",
        "corpus_path": str(CORPUS_V1_PATH.relative_to(PROJECT_ROOT)), "corpus_commit": CORPUS_V1_COMMIT,
        "corpus_sha256": sha256(CORPUS_V1_PATH),
        "benchmark_sha256": sha256(graph_common.BENCHMARK_PATH),
        "clause_graph_sha256": sha256(graph_common.CLAUSE_GRAPH_PATH),
        "query_mode": "question", "embedding_model": MODEL_NAME, "query_prefix": BGE_PREFIX,
        "lexical_mode": LEXICAL_MODE, "seeds": list(SEEDS), "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
        "validation_fold": "(test_fold + 1) mod 5", "folds": folds,
        "models": {m: MODELS[m] for m in models},
        "train_config": asdict(TrainConfig(kind="<per model>", relation_set="<per model>")),
        "relations": rel_info, "n_edges": n_edges, "graphexp_defaults": dict(GRAPHEXP_DEFAULTS),
        "clause_prior": "gold frequency in the 4 non-test folds; ties by dense score, then clause ID",
        "n_training_runs": n_runs,
    })
    print("\n".join(md))


if __name__ == "__main__":
    main()
