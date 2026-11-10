"""Compare a real baseline run with the frozen operator-gloss diagnostic.

Standard library only; no models, score tuning or canonical data changes.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baseline/specgraph_baselines"))
import retrieval_baselines as baseline


def verify(run_dir: Path, diagnostic_dir: Path, write_report: bool = True) -> dict:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    reference_config = json.loads((diagnostic_dir / "run_config.json").read_text(encoding="utf-8"))
    for key in ("chunks", "queries"):
        if config["input_sha256"][key] != reference_config["input_sha256"][key]:
            raise ValueError(f"Different {key}: this is not the same paired experiment")
    for key, expected in (("query_transform", "operator_gloss"), ("query_mode", "query_code"),
                          ("query_transform_version", "operator_gloss_v1"),
                          ("lexical_mode", "content"), ("dense_backend", "exact"),
                          ("query_prefix", reference_config["query_prefix"]),
                          ("dense_weight", 2), ("rrf_k", 60), ("effective_candidate_k", 50),
                          ("embedding_model", reference_config["embedding_model"]),
                          ("embedding_revision", reference_config["embedding_revision"])):
        if config.get(key) != expected:
            raise ValueError(f"Unexpected configuration: {key}={config.get(key)!r}")
    for key, filename in (("script", "retrieval_baselines.snapshot.py"),
                          ("query_transforms", "query_transforms.snapshot.py")):
        if config["input_sha256"][key] != baseline.file_sha(run_dir / filename):
            raise ValueError(f"Source snapshot hash mismatch: {filename}")
    if config["query_inputs_sha256"] != baseline.file_sha(run_dir / "query_inputs.jsonl"):
        raise ValueError("Exact query input hash mismatch")
    original_queries = baseline.read_records(ROOT / "data/benchmark/spec_benchmark_v1.jsonl")
    if baseline.file_sha(ROOT / "data/benchmark/spec_benchmark_v1.jsonl") != config["input_sha256"]["queries"]:
        raise ValueError("Current benchmark no longer matches this run")
    chunks_path = ROOT / "data/processed/chunks.jsonl"
    if baseline.file_sha(chunks_path) != config["input_sha256"]["chunks"]:
        raise ValueError("Current corpus no longer matches this run")
    chunks = {c["chunk_id"]: c for c in baseline.read_records(chunks_path)}
    expected_inputs = {r["query_id"]: r["text"] for r in baseline.read_records(diagnostic_dir / "query_inputs.jsonl")
                       if r["variant"] == "operator_gloss"}
    actual_inputs = baseline.read_records(run_dir / "query_inputs.jsonl")
    if len(actual_inputs) != len(expected_inputs) or any(set(r) != {"query_id", "text"} for r in actual_inputs):
        raise ValueError("Input count/schema mismatch or annotations exported to query inputs")
    if {r["query_id"]: r["text"] for r in actual_inputs} != expected_inputs:
        raise ValueError("Input strings differ from the frozen diagnostic")
    reference = {r["query_id"]: r for r in baseline.read_records(diagnostic_dir / "ranked_results.jsonl")
                 if r["variant"] == "operator_gloss"}
    actual = baseline.read_records(run_dir / "ranked_results.jsonl")
    if len(actual) != len(reference) or {r["query_id"] for r in actual} != set(reference):
        raise ValueError("Ranking query IDs differ")
    gold = {q["query_id"]: q["gold_clause_ids"] for q in original_queries}
    methods = {"bm25": "bm25", "dense": "dense", "hybrid_rrf": "hybrid_2_1"}
    matched = {m: {"top_10": 0, "top_50": 0} for m in methods}
    differences = []
    for row in actual:
        qid = row["query_id"]
        if row["gold_clause_ids"] != gold[qid]:
            raise ValueError(f"Gold changed: {qid}")
        for method, reference_method in methods.items():
            ranked = row["results"][method]
            expected = reference[qid]["results"][reference_method]
            evidence = row["evidence_chunk_ids"][method]
            if len(ranked) != 50 or len(set(ranked)) != 50 or len(evidence) != 50:
                raise ValueError(f"Invalid clause/evidence budget: {qid}/{method}")
            if any(chunks[cid]["clause_id"] != clause for cid, clause in zip(evidence, ranked)):
                raise ValueError(f"Invalid evidence mapping: {qid}/{method}")
            for k in (10, 50):
                matched[method][f"top_{k}"] += ranked[:k] == expected[:k]
            if ranked != expected:
                differences.append({"query_id": qid, "method": method,
                                    "first_different_rank": next(i + 1 for i, (a, b) in enumerate(zip(ranked, expected)) if a != b)})
    stored_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if stored_metrics != baseline.aggregate(actual, config["methods"], sorted(set(config["top_k"]))):
        raise ValueError("Stored metrics do not match saved rankings")
    report = {"status": "pass" if not differences else "rank_differences",
              "queries": len(actual), "exact_input_matches": len(actual_inputs),
              "rank_matches": matched, "rank_differences": differences,
              "gold_labels_unchanged": True, "evidence_mappings_valid": True,
              "stored_metrics_verified": True,
              "input_sha256": {key: config["input_sha256"][key] for key in ("chunks", "queries")},
              "baseline_config_sha256": baseline.file_sha(run_dir / "run_config.json"),
              "diagnostic_config_sha256": baseline.file_sha(diagnostic_dir / "run_config.json"),
              "verifier_sha256": baseline.file_sha(__file__)}
    if write_report:
        (run_dir / "reproduction_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/baseline/operator_gloss")
    parser.add_argument("--diagnostic-dir", type=Path, default=ROOT / "results/query_diagnostics")
    parser.add_argument("--check-only", action="store_true", help="Verify without overwriting the historical verification record")
    args = parser.parse_args()
    result = verify(args.run_dir, args.diagnostic_dir, write_report=not args.check_only)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
