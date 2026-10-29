"""Replay a fixed weighted-RRF grid from saved BM25 and dense clause rankings.

This requires no model inference. Each original retriever contributes its same
saved candidate list; missing clauses contribute zero to that branch's score.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from retrieval_baselines import aggregate, file_sha, read_records, reciprocal_rank_fusion

DENSE_WEIGHTS = (1, 2, 3)


def replay(source: Path, output: Path):
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    rows = read_records(source / "ranked_results.jsonl")
    limit, rrf_k = config["effective_candidate_k"], config["rrf_k"]
    output_rows = []
    originals = [m for m in config["methods"] if m != "hybrid_rrf"]
    methods = ["bm25", "dense"] + [f"hybrid_rrf_d{w}" for w in DENSE_WEIGHTS]
    methods += [m for m in originals if m not in methods]
    for original in rows:
        row = {**original, "results": {m: list(original["results"][m]) for m in originals},
               "evidence_chunk_ids": {m: list(original["evidence_chunk_ids"][m]) for m in originals}}
        for weight in DENSE_WEIGHTS:
            method = f"hybrid_rrf_d{weight}"
            ranking = reciprocal_rank_fusion([original["results"]["bm25"], original["results"]["dense"]],
                                              limit, rrf_k, [1.0, float(weight)])
            if weight == 1 and config.get("dense_weight", 1) == 1:
                if ranking != original["results"]["hybrid_rrf"]:
                    raise ValueError("Equal-weight replay does not match the saved hybrid ranking")
            row["results"][method] = ranking
            # Original full chunk rankings were not saved. Carry a real source
            # evidence chunk from the branch with the highest clause-rank
            # contribution, with dense winning ties. This does not affect the
            # clause retrieval metrics; no cross-encoder uses replayed evidence.
            choices = {}
            for branch, branch_weight in (("dense", weight), ("bm25", 1)):
                for rank, (cid, eid) in enumerate(zip(original["results"][branch],
                                                     original["evidence_chunk_ids"][branch]), 1):
                    contribution = branch_weight / (rrf_k + rank)
                    if cid not in choices or contribution > choices[cid][0]:
                        choices[cid] = (contribution, eid)
            row["evidence_chunk_ids"][method] = [choices[cid][1] for cid in ranking]
        if "node_results" in original:
            nodes = {cid: nid for m, ranking in original["results"].items()
                     for cid, nid in zip(ranking, original["node_results"][m])}
            row["node_results"] = {m: [nodes[cid] for cid in row["results"][m]] for m in methods}
        output_rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ranked_results.jsonl").open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = aggregate(output_rows, methods, [5, 10, 50])
    derived = {**config, "source_run": str(source.resolve()), "output_dir": str(output.resolve()),
               "methods": methods, "query_mode": config.get("query_mode", "query_code"),
               "fusion_weights_by_method": {f"hybrid_rrf_d{w}": {"bm25": 1, "dense": w} for w in DENSE_WEIGHTS},
               "replay_script_sha256": file_sha(__file__),
               "source_rankings_sha256": file_sha(source / "ranked_results.jsonl"),
               "source_config_sha256": file_sha(source / "run_config.json"),
               "replayed_at_utc": datetime.now(timezone.utc).isoformat(),
               "replayed_evidence_policy": "highest weighted clause-rank contribution; dense wins ties",
               "reranker_note": "Any cross-encoder result is copied from the original equal-weight run, not rerun for new weights"}
    for name, obj in (("metrics.json", metrics), ("run_config.json", derived)):
        (output / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    for name in ("corpus_manifest.jsonl", "data_audit.json", "model_revisions.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    print(f"Saved fusion grid: {output}")
    for method, values in metrics["overall"].items():
        print(method, "recall@10=", values["recall@10"], "coverage@10=", values["complete_coverage@10"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output_dir.resolve():
        parser.error("Use a different output directory to preserve original results")
    replay(args.source, args.output_dir)


if __name__ == "__main__":
    main()
