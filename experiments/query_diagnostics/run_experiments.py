"""Paired query-representation diagnostics with fixed text/graph retrievers.

Does not edit benchmark labels or existing milestone artifacts. Run from root:
  baseline/.venv/Scripts/python.exe experiments/query_diagnostics/run_experiments.py
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline/specgraph_baselines"))
sys.path.insert(0, str(ROOT / "graph/specgraph_graph"))
import retrieval_baselines as baseline
from clause_graph import DEFAULT_EDGE_TYPES, adjacency, load_clause_graph, node_ids
from clause_scores import chunk_to_clause_max, load_scores
from evaluate import paired
from graph_expand import graph_expand, normalized_adjacency, rank
from query_variants import CONTROLS, VARIANTS, build_variants

METHODS = ("bm25", "dense", "hybrid_2_1", "dense_graph_exp")
KS = (1, 3, 5, 10)
METRICS = ("recall@5", "complete_coverage@5", "recall@10", "complete_coverage@10", "ndcg@10")
CUE_RE = re.compile(r"prefix|postfix|increment|decrement|shift|subscript|assignment|conditional|comma|logical|sequenc|prvalue|bool|enumeration|overload", re.I)
VAGUE_RE = re.compile(r"\b(this|these|here|inputs?|statement|updating|modifying)\b", re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def features(query):
    q, code = query["query"], query["code"]
    return {"query_id": query["query_id"], "difficulty": query["difficulty"],
            "n_gold": len(query["gold_clause_ids"]), "question_words": len(q.split()),
            "code_chars": len(code), "question_marks": q.count("?"),
            "domain_cue": bool(CUE_RE.search(q)), "vague_reference": bool(VAGUE_RE.search(q))}


def summarize(rows):
    return {metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}


def analyze(all_rows, query_features):
    lookup = {(r["variant"], r["method"], r["query_id"]): r for r in all_rows}
    overall, contrasts, groups, graph_pairs = {}, [], [], []
    qids = list(query_features)
    for variant in VARIANTS:
        overall[variant] = {}
        for method in METHODS:
            subset = [lookup[variant, method, qid] for qid in qids]
            overall[variant][method] = summarize(subset)
            if variant in CONTROLS:
                control = CONTROLS[variant]
                for metric in METRICS:
                    result = paired([r[metric] for r in subset],
                                    [lookup[control, method, qid][metric] for qid in qids])
                    contrasts.append({"variant": variant, "control": control, "method": method,
                                      "metric": metric, "n": len(qids), **result})
        for metric in ("recall@10", "complete_coverage@10"):
            graph_pairs.append({"variant": variant, "metric": metric,
                                **paired([lookup[variant, "dense_graph_exp", qid][metric] for qid in qids],
                                         [lookup[variant, "dense", qid][metric] for qid in qids])})
    # Observational groups for the original question-only representation, with
    # gold-size strata so lexical cue and task complexity are not conflated.
    for method in METHODS:
        for stratum in ("all", "gold_1", "gold_2", "gold_3_plus"):
            eligible = [qid for qid, f in query_features.items()
                        if stratum == "all" or (stratum == "gold_1" and f["n_gold"] == 1)
                        or (stratum == "gold_2" and f["n_gold"] == 2)
                        or (stratum == "gold_3_plus" and f["n_gold"] >= 3)]
            for feature in ("domain_cue", "vague_reference", "question_words_le12", "code_chars_le80"):
                for value in (False, True):
                    members = []
                    for qid in eligible:
                        f = query_features[qid]
                        flag = (f["question_words"] <= 12 if feature == "question_words_le12" else
                                f["code_chars"] <= 80 if feature == "code_chars_le80" else f[feature])
                        if flag == value:
                            members.append(lookup["question_only", method, qid])
                    if members:
                        groups.append({"method": method, "stratum": stratum, "feature": feature,
                                       "value": value, "n": len(members), **summarize(members)})
    return {"overall": overall, "paired_contrasts": contrasts, "observational_groups": groups,
            "graph_vs_dense": graph_pairs}


def report(output, analysis, query_features, queries, rewrites, all_rows):
    overall = analysis["overall"]
    pct = lambda x: f"{100*x:.2f}"
    lines = ["# Query Representation and Dataset Diagnostics\n",
             "This is a fixed, paired diagnostic experiment on all 49 existing queries, not a new benchmark or a held-out test. "
             "Every natural-language source question already consists of a single question sentence (one final question mark, no internal sentence boundary). "
             "Consequently, the existing benchmark cannot establish that one-sentence queries are better than multi-sentence queries. "
             "The experiment separates information removal, formatting, explicitness, and verbosity as far as the available data allow.\n",
             "## Design\n",
             "Nine representations are evaluated with fixed BM25, Dense, 2:1 Hybrid RRF, and Dense + GraphExp. "
             "BGE-small, corpus text, clause aggregation, candidate budget 50, and graph parameters "
             "(top-5 seeds, one hop, lambda=0.5, references+parent+sibling) are unchanged. "
             "The original benchmark, graph, gold labels, and milestone results are not modified.\n",
             "| Variant | What changes | Primary comparison |\n|---|---|---|",
             "| question_only | Original question; no code | Reference |",
             "| question_code | Original question followed by original code | Information added vs question_only |",
             "| code_only | Original code; no question | Information removed vs question_code |",
             "| labeled_code | Same question and code, explicit labels and code fence | Formatting vs question_code |",
             "| code_first | Same labeled fields in reverse order | Order vs labeled_code |",
             "| explicit_one_sentence | Authored, operator-specific question plus unchanged code | Explicitness vs question_code |",
             "| verbose_explicit | Same explicit question/code plus generic preamble and instruction | Verbosity/framing vs explicit_one_sentence |",
             "| operator_gloss | Original question/code plus deterministic operator-spelling glossary | Symbol verbalization vs question_code |",
             "| question_domain | C++23 specification prefix plus original question; no code | Domain framing vs question_only |\n",
             "The 49 authored explicit questions are candidate paraphrases derived from the visible question and code. "
             "They contain no clause IDs, gold lists, expected-property metadata, or retrieved evidence. "
             "They have not received independent semantic review: extra operator-specific wording can narrow an underspecified task. "
             "Retained labels are therefore provisional for those two rewrite conditions. They must not silently replace canonical benchmark questions. "
             "Verbose framing changes both length and instruction context; it is not a pure causal test of sentence count.\n",
             "## Overall Results\n",
             "Recall/coverage are percentages. Coverage means every gold clause is present.\n",
             "| Variant | Method | R@5 | Cov@5 | R@10 | Cov@10 | nDCG@10 |\n|---|---|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        for method in METHODS:
            r = overall[variant][method]
            lines.append(f"| {variant} | {method} | {pct(r['recall@5'])} | {pct(r['complete_coverage@5'])} | "
                         f"{pct(r['recall@10'])} | {pct(r['complete_coverage@10'])} | {r['ndcg@10']:.3f} |")
    lines += ["\n## Paired Changes in Dense Retrieval\n",
              "Per-query differences, with 95% percentile bootstrap intervals (2,000 query resamples, seed 0). "
              "Intervals are exploratory and not adjusted for multiple comparisons; variants are not independent datasets.\n",
              "| Variant vs control | Metric | Mean change (pp) | 95% interval (pp) | Wins/losses/ties |\n|---|---|---:|---|---|"]
    for r in analysis["paired_contrasts"]:
        if r["method"] == "dense" and r["metric"] in ("recall@10", "complete_coverage@10"):
            lo, hi = r["ci95"]
            lines.append(f"| {r['variant']} vs {r['control']} | {r['metric']} | {100*r['mean_diff']:+.2f} | "
                         f"[{100*lo:+.2f}, {100*hi:+.2f}] | {r['wins']}/{r['losses']}/{r['ties']} |")
    lines += ["\n## Original-Query Features and Confounding\n",
              "These are observational subgroup summaries, not effects of changing query text. The 'domain cue' regex "
              "detects operator/type/sequence vocabulary; 'vague reference' detects words such as this, here, input, "
              "statement, updating, or modifying. These transparent heuristics are imperfect and can overlap. "
              "Length cutoffs are 12 question words and 80 code characters. Gold-size strata are reported to separate "
              "surface wording from the number of required clauses. Small groups should not be generalized.\n",
              "| Gold stratum | Feature | Present | n | Dense R@10 | Dense Cov@10 |\n|---|---|---|---:|---:|---:|"]
    for r in analysis["observational_groups"]:
        if r["method"] == "dense" and r["feature"] in ("domain_cue", "question_words_le12"):
            lines.append(f"| {r['stratum']} | {r['feature']} | {r['value']} | {r['n']} | {pct(r['recall@10'])} | {pct(r['complete_coverage@10'])} |")
    lines += ["\n## Does the Graph Benefit Differ by Query Representation?\n",
              "Same graph and propagation parameters for every variant; no graph setting was chosen from these results.\n",
              "| Variant | Graph minus Dense Cov@10 (pp) | Wins/losses/ties |\n|---|---:|---|"]
    for r in analysis["graph_vs_dense"]:
        if r["metric"] == "complete_coverage@10":
            lines.append(f"| {r['variant']} | {100*r['mean_diff']:+.2f} | {r['wins']}/{r['losses']}/{r['ties']} |")
    lines += ["\n## Illustrative Rewrite Cases\n",
              "The three largest positive and three largest negative Dense Recall@10 changes for explicit rewrites "
              "are selected after evaluation to expose both directions, not as representative samples.\n"]
    lookup = {(r["variant"], r["method"], r["query_id"]): r for r in all_rows}
    deltas = sorted((lookup["explicit_one_sentence", "dense", q["query_id"]]["recall@10"] -
                     lookup["question_code", "dense", q["query_id"]]["recall@10"], q["query_id"]) for q in queries)
    by_id = {q["query_id"]: q for q in queries}
    for delta, qid in dict.fromkeys(deltas[:3] + deltas[-3:]):
        q = by_id[qid]
        lines += [f"- **{qid}**, {len(q['gold_clause_ids'])} gold clauses, change {100*delta:+.2f} pp. "
                  f"Original: {q['query']} Rewrite: {rewrites[qid]}"]
    lines += ["\n## Dataset Improvement Decisions\n",
              "1. Preserve the original task and code, and add reviewed alternative wording as a separate field. "
              "Require operator/type/reference clarity where essential, without naming gold clauses or revealing the outcome.\n",
              "2. Build matched concise/verbose and plain-language/technical-term pairs. Keep code, assumptions, and intended evidence fixed. "
              "Have a reviewer check semantic equivalence and evidence sufficiency before using the same gold set.\n",
              "3. Add code-dependent contrast sets: similar questions with different operators or types. Retain both easy and difficult cases. "
              "A question-only retriever can benefit from linguistic cues without demonstrating that it distinguishes the programs.\n",
              "4. Expand beyond the 11 currently gold-labeled clauses and annotate the supporting paragraph or span. "
              "Complete clause coverage can count a non-supporting paragraph as correct, so it is not a full evidence-quality check.\n",
              "5. Group all paraphrases and related code families into the same train/dev/test split. "
              "Select retrieval/query-processing settings on dev; report original, variant-average, and worst-variant results on frozen test. "
              "Do not count nine versions of 49 questions as 441 independent test items.\n",
              "## Reproduction\n",
              "Run `baseline/.venv/Scripts/python.exe experiments/query_diagnostics/run_experiments.py` from the project root. "
              "Weights are loaded offline from the existing cache at the recorded model revision. "
              "Document embeddings are cached under `baseline/.cache/query_diagnostics/`. "
              "Use `--analyze-only` to regenerate this report from saved per-query metrics, without models.\n",
              "`experiment_plan.json` records the fixed conditions and source hashes; `query_inputs.jsonl` contains exact retrieval texts; "
              "`clause_scores.json` stores full scores; `ranked_results.jsonl` stores top-50 clauses; `per_query.csv` contains metrics/features; "
              "`analysis.json` contains all paired intervals and strata; `run_config.json` records package versions and cache checks. "
              "Existing milestone artifacts and benchmark records remain unchanged.\n"]
    (output / "RESULTS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/query_diagnostics")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    benchmark = ROOT / "data/benchmark/spec_benchmark_v1.jsonl"
    corpus = ROOT / "data/processed/chunks.jsonl"
    queries = baseline.read_records(benchmark)
    rewrites_path = Path(__file__).with_name("explicit_questions.json")
    rewrites = json.loads(rewrites_path.read_text(encoding="utf-8"))
    assert set(rewrites) == {q["query_id"] for q in queries}
    query_features = {q["query_id"]: features(q) for q in queries}
    if args.analyze_only:
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        if config["input_sha256"]["queries"] != sha(benchmark) or config["input_sha256"]["rewrites"] != sha(rewrites_path):
            raise ValueError("Saved run no longer matches current benchmark/rewrites")
        with (output / "per_query.csv").open(encoding="utf-8") as stream:
            all_rows = list(csv.DictReader(stream))
        for row in all_rows:
            for metric in METRICS:
                row[metric] = float(row[metric])
        analysis = analyze(all_rows, query_features)
        write_json(output / "analysis.json", analysis)
        report(output, analysis, query_features, queries, rewrites, all_rows)
        print("Analysis/report regenerated without inference.")
        return
    graph_path = ROOT / "artifacts/graph/clause_graph.json"
    hashes = {"queries": sha(benchmark), "chunks": sha(corpus), "rewrites": sha(rewrites_path),
              "graph": sha(graph_path), "runner": sha(__file__), "variants": sha(Path(__file__).with_name("query_variants.py")),
              "retrieval_module": sha(baseline.__file__)}
    plan = {"variants": VARIANTS, "controls": CONTROLS, "methods": METHODS, "ks": KS,
            "candidate_k": 50, "dense_weight": 2, "rrf_k": 60,
            "graph": {"seed_k": 5, "hops": 1, "lam": 0.5, "edge_types": DEFAULT_EDGE_TYPES},
            "rewrite_review_status": "assistant-authored candidate rewrites; independent semantic review pending",
            "bootstrap_samples": 2000, "bootstrap_seed": 0, "input_sha256": hashes}
    write_json(output / "experiment_plan.json", plan)
    print("Fixed plan saved before inference.", flush=True)
    inputs, metadata = {}, []
    for q in queries:
        texts = build_variants(q["query"], q["code"], rewrites[q["query_id"]])
        for variant in VARIANTS:
            inputs[variant, q["query_id"]] = texts[variant]
            metadata.append({"query_id": q["query_id"], "variant": variant, "text": texts[variant]})
    with (output / "query_inputs.jsonl").open("w", encoding="utf-8") as stream:
        for row in metadata:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.environ.setdefault("HF_HOME", str(ROOT / "baseline/.cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    torch.set_num_threads(args.threads)
    from sentence_transformers import SentenceTransformer
    model_name = baseline.DEFAULT_EMBEDDING
    revisions = json.loads((ROOT / "results/baseline/summary/model_revisions.json").read_text())
    model = SentenceTransformer(model_name, revision=revisions[model_name])
    prefix = "Represent this sentence for searching relevant passages: "
    chunks = [baseline.normalize_chunk(c) for c in baseline.read_records(corpus)]
    docs = [baseline.document_text(c) for c in chunks]
    cache = ROOT / "baseline/.cache/query_diagnostics"
    cache.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps([hashes["chunks"], model_name, revisions[model_name],
                                         model.max_seq_length, version("sentence-transformers")]).encode()).hexdigest()
    cache_file = cache / f"documents_{signature[:16]}.npy"
    if cache_file.exists():
        vectors = np.load(cache_file, allow_pickle=False)
        print("Reusing cached document embeddings.", flush=True)
    else:
        vectors = model.encode_document(docs, normalize_embeddings=True, show_progress_bar=True)
        np.save(cache_file, vectors)
    assert vectors.shape[0] == len(chunks)
    texts = [row["text"] for row in metadata]
    qvectors = model.encode_query(texts, prompt=prefix, normalize_embeddings=True, show_progress_bar=True)
    graph = load_clause_graph(graph_path)
    ids = node_ids(graph)
    chunk_clauses = [c["clause_id"] for c in chunks]
    bm25 = baseline.BM25([baseline.lexical_tokens(t, "content") for t in docs])
    a_norm = normalized_adjacency(adjacency(graph, DEFAULT_EDGE_TYPES))
    cached = load_scores()
    assert ids == cached["clause_ids"] and [q["query_id"] for q in queries] == cached["query_ids"]
    saved = {r["query_id"]: r for r in baseline.read_records(ROOT / "results/baseline/question_only/ranked_results.jsonl")}
    scores = {v: {"dense": [], "bm25": []} for v in VARIANTS}
    rankings, all_rows = [], []
    baseline_checks = 0
    for row_index, item in enumerate(metadata):
        qid, variant = item["query_id"], item["variant"]
        query_index = row_index // len(VARIANTS)
        q = queries[query_index]
        d = np.round(chunk_to_clause_max(vectors @ qvectors[row_index], chunk_clauses, ids), 7)
        lexical = np.round(chunk_to_clause_max(np.array(bm25.get_scores(baseline.lexical_tokens(item["text"], "content"))), chunk_clauses, ids), 6)
        if variant == "question_only":
            if rank(d, ids)[:50] != saved[qid]["results"]["dense"] or rank(lexical, ids)[:50] != saved[qid]["results"]["bm25"]:
                raise ValueError(f"Reference ranking reproduction failed: {qid}")
            # Keep the teammate's exact cached reference vectors for comparability.
            d, lexical = cached["dense"][query_index], cached["bm25"][query_index]
            baseline_checks += 1
        dense_rank, bm25_rank = rank(d, ids), rank(lexical, ids)
        result = {"bm25": bm25_rank[:50], "dense": dense_rank[:50],
                  "hybrid_2_1": baseline.reciprocal_rank_fusion([bm25_rank[:50], dense_rank[:50]], 50, 60, [1, 2]),
                  "dense_graph_exp": rank(graph_expand(d, a_norm, seed_k=5, hops=1, lam=0.5), ids)[:50]}
        for method, ranking in result.items():
            values = {key: value for k in KS for key, value in baseline.metrics_at_k(ranking, q["gold_clause_ids"], k).items()}
            all_rows.append({"variant": variant, "method": method, **query_features[qid],
                             "input_words": len(item["text"].split()), **values, "top10": ";".join(ranking[:10])})
        rankings.append({"variant": variant, "query_id": qid, "results": result})
        scores[variant]["dense"].append(d.tolist())
        scores[variant]["bm25"].append(lexical.tolist())
    # Input metadata is query-major; every variant array retains benchmark order.
    write_json(output / "clause_scores.json", {"clause_ids": ids, "query_ids": cached["query_ids"], "scores": scores})
    with (output / "ranked_results.jsonl").open("w", encoding="utf-8") as stream:
        for row in rankings:
            stream.write(json.dumps(row) + "\n")
    with (output / "per_query.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    analysis = analyze(all_rows, query_features)
    write_json(output / "analysis.json", analysis)
    config = {**plan, "created_at_utc": datetime.now(timezone.utc).isoformat(), "threads": args.threads,
              "embedding_model": model_name, "embedding_revision": revisions[model_name], "query_prefix": prefix,
              "packages": {p: version(p) for p in ("torch", "numpy", "sentence-transformers", "transformers")},
              "reference_rankings_verified": baseline_checks, "document_embedding_sha256": sha(cache_file),
              "reference_scores_sha256": sha(ROOT / "artifacts/scores/question_scores.json"),
              "document_truncation": baseline.truncation_report(model, docs),
              "query_truncation": baseline.truncation_report(model, [prefix + t for t in texts]),
              "query_count": len(queries), "representations_per_query": len(VARIANTS)}
    write_json(output / "run_config.json", config)
    report(output, analysis, query_features, queries, rewrites, all_rows)
    assert sha(benchmark) == hashes["queries"] and sha(corpus) == hashes["chunks"]
    print("Finished. 49 reference rankings reproduced; canonical inputs unchanged.", flush=True)
    for variant in VARIANTS:
        r = analysis["overall"][variant]["dense"]
        print(variant, "Dense R@10", round(100*r['recall@10'], 2), "Cov@10", round(100*r['complete_coverage@10'], 2))


if __name__ == "__main__":
    main()
