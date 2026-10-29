"""Compare saved baseline runs on the same benchmark; no model execution."""
import argparse
import csv
import json
from pathlib import Path

from retrieval_baselines import aggregate, read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "preliminary_summary")
    args = parser.parse_args()
    comparison, flat_rows, signatures, annotations = [], [], set(), None
    for directory in args.runs:
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        signatures.add((config["input_sha256"]["chunks"], config["input_sha256"]["queries"]))
        if len(signatures) > 1:
            raise ValueError("Comparison requires identical corpus and benchmark files")
        rows = read_records(directory / "ranked_results.jsonl")
        current = {row["query_id"]: row["gold_clause_ids"] for row in rows}
        if annotations is not None and annotations != current:
            raise ValueError("Comparison requires the same evaluated queries and labels")
        annotations = current
        methods = config["methods"]
        # All chosen runs use >=50 candidates, allowing candidate recall@50.
        if any(len(row["results"][method]) < 50 for row in rows for method in methods):
            raise ValueError("Expected at least 50 candidates per query/method")
        metrics = aggregate(rows, methods, [5, 10, 50])
        comparison.append({"run": directory.name, "directory": str(directory.resolve()),
                           "configuration": config, "metrics": metrics})
        for method in methods:
            flat_rows.append({"run": directory.name, "method": method,
                              "query_mode": config.get("query_mode", "query_code"),
                              "dense_to_bm25_weight": config.get("fusion_weights_by_method", {}).get(method, {}).get("dense", ""),
                              "lexical_mode": config.get("lexical_mode", "raw"),
                              "embedding_model": config["embedding_model"] if method != "bm25" else "",
                              "queries": metrics["evaluated_queries"],
                              **metrics["overall"][method]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(flat_rows, indent=2))


if __name__ == "__main__":
    main()
