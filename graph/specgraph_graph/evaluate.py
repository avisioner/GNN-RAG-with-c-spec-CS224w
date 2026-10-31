"""Step 4 metrics. Per-query metrics come from Erica's `metrics_at_k`, so every
method (hers and graph expansion) goes through one definition:

    recall@k            |top-k ∩ gold| / |gold|
    precision@k         |top-k ∩ gold| / k
    complete_coverage@k 1 if gold ⊆ top-k else 0
    ndcg@k              binary-relevance nDCG

Rankings are distinct clause IDs. Averages are macro over queries.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from common import erica

KS = (1, 3, 5, 10)
BOOT_SAMPLES = 2000
BOOT_SEED = 0


def per_query_metrics(ranking: Sequence[str], gold: Sequence[str], ks: Iterable[int] = KS) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out.update(erica.metrics_at_k(list(ranking), list(gold), k))
    return out


def evaluate_method(rankings: dict[str, Sequence[str]], queries: list[dict[str, Any]],
                    ks: Iterable[int] = KS) -> dict[str, dict[str, float]]:
    """query_id -> metric dict."""
    return {q["query_id"]: per_query_metrics(rankings[q["query_id"]], q["gold"], ks) for q in queries}


def mean_by_group(per_query: dict[str, dict[str, float]], queries: list[dict[str, Any]],
                  key: Callable[[dict[str, Any]], Any] | None = None) -> dict[Any, dict[str, float]]:
    """Macro averages, overall (key None -> group 'all') or per group."""
    groups: dict[Any, list[str]] = defaultdict(list)
    for q in queries:
        groups["all" if key is None else key(q)].append(q["query_id"])
    out = {}
    for g, qids in groups.items():
        names = per_query[qids[0]].keys()
        out[g] = {n: float(np.mean([per_query[qid][n] for qid in qids])) for n in names}
        out[g]["n"] = len(qids)
    return out


def boot_ci(values: Sequence[float], n_boot: int = BOOT_SAMPLES, seed: int = BOOT_SEED,
            alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap over queries, resampled with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Per-query wins/losses/ties of a vs b, plus a bootstrap CI of mean(a - b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mean, lo, hi = boot_ci(a - b)
    return {"wins": int((a > b).sum()), "losses": int((a < b).sum()), "ties": int((a == b).sum()),
            "mean_diff": mean, "ci95": (lo, hi)}


def reachability(seed_ids: Sequence[str], gold: Sequence[str], adj: np.ndarray,
                 ids: Sequence[str]) -> dict[str, float]:
    """How selective a 1-hop expansion from `seed_ids` is for one query.

    gold_in_reach   fraction of gold in seeds ∪ N(seeds)
    chance_reach    fraction of all clauses in seeds ∪ N(seeds)
    missed_found    fraction of gold *not* in seeds that are in N(seeds)  (nan if none missed)
    chance_missed   fraction of non-seed clauses that are in N(seeds)
    """
    index = {c: i for i, c in enumerate(ids)}
    seeds = np.zeros(len(ids), dtype=bool)
    seeds[[index[s] for s in seed_ids]] = True
    neigh = (adj[seeds].sum(axis=0) > 0) & ~seeds
    reach = seeds | neigh
    g = np.array([index[x] for x in gold])
    missed = g[~seeds[g]]
    return {"gold_in_reach": float(reach[g].mean()), "chance_reach": float(reach.mean()),
            "missed_found": float(neigh[missed].mean()) if len(missed) else float("nan"),
            "chance_missed": float(neigh.sum() / (~seeds).sum()), "n_missed": int(len(missed))}
