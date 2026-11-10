"""Check repository links, relocated run hashes and saved evaluation artifacts.

Uses the standard library only; does not load models or rewrite any results.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline/specgraph_baselines"))
import retrieval_baselines as baseline


def main() -> None:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    links = 0
    for name in sorted(set(paths)):
        path = ROOT / name
        if not name or not path.is_file() or path.suffix.lower() != ".md":
            continue
        for match in re.finditer(r"\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", path.read_text(encoding="utf-8-sig")):
            target = match.group(1).strip("<>")
            if urlparse(target).scheme or target.startswith("#"):
                continue
            local = unquote(target.split("#", 1)[0])
            if not (path.parent / local).exists():
                raise ValueError(f"Broken link in {name}: {target}")
            links += 1

    manifest = json.loads((ROOT / "results/layout_manifest.json").read_text(encoding="utf-8-sig"))
    for row in manifest["preserved_files"]:
        if baseline.file_sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Relocated artifact changed: {row['path']}")
    for old in manifest["relocations"]:
        if (ROOT / old).exists():
            raise ValueError(f"Obsolete duplicate path remains: {old}")

    queries = baseline.read_records(ROOT / "data/benchmark/spec_benchmark_v1.jsonl")
    gold = {q["query_id"]: q["gold_clause_ids"] for q in queries}
    chunks = {c["chunk_id"]: c for c in baseline.read_records(ROOT / "data/processed/chunks.jsonl")}
    clause_ids = {c["clause_id"] for c in chunks.values()}
    run_count = 0
    for directory in sorted((ROOT / "results/baseline").iterdir()):
        if not directory.is_dir() or not (directory / "metrics.json").exists():
            continue
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        for key, source in (("chunks", "data/processed/chunks.jsonl"),
                            ("queries", "data/benchmark/spec_benchmark_v1.jsonl")):
            if config["input_sha256"][key] != baseline.file_sha(ROOT / source):
                raise ValueError(f"Input mismatch: {directory.name}/{key}")
        rows = baseline.read_records(directory / "ranked_results.jsonl")
        if len(rows) != len(gold) or {r["query_id"] for r in rows} != set(gold):
            raise ValueError(f"Query IDs mismatch: {directory.name}")
        for row in rows:
            if row["gold_clause_ids"] != gold[row["query_id"]]:
                raise ValueError(f"Changed gold: {directory.name}/{row['query_id']}")
            for method, ranking in row["results"].items():
                evidence = row["evidence_chunk_ids"][method]
                if len(ranking) != 50 or len(set(ranking)) != 50 or len(evidence) != 50:
                    raise ValueError(f"Invalid candidates: {directory.name}/{method}")
                if not set(ranking) <= clause_ids:
                    raise ValueError(f"Unknown clause: {directory.name}/{method}")
                if any(chunks[e]["clause_id"] != cid for e, cid in zip(evidence, ranking)):
                    raise ValueError(f"Invalid evidence: {directory.name}/{method}")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        first = next(iter(metrics["overall"].values()))
        ks = sorted(int(key.split("@")[1]) for key in first if key.startswith("recall@"))
        if baseline.aggregate(rows, config["methods"], ks) != metrics:
            raise ValueError(f"Metrics do not match rankings: {directory.name}")
        for key, filename in (("script", "retrieval_baselines.snapshot.py"),
                              ("query_transforms", "query_transforms.snapshot.py")):
            if (directory / filename).exists() and baseline.file_sha(directory / filename) != config["input_sha256"][key]:
                raise ValueError(f"Snapshot hash mismatch: {directory.name}/{filename}")
        run_count += 1

    diagnostic = ROOT / "results/query_diagnostics"
    inputs = baseline.read_records(diagnostic / "query_inputs.jsonl")
    rankings = baseline.read_records(diagnostic / "ranked_results.jsonl")
    keys = {(r["query_id"], r["variant"]) for r in inputs}
    if len(inputs) != len(queries) * 9 or len(keys) != len(inputs) or len(rankings) != len(inputs):
        raise ValueError("Missing or duplicate paired diagnostic inputs/rankings")
    if keys != {(r["query_id"], r["variant"]) for r in rankings}:
        raise ValueError("Diagnostic input/ranking IDs differ")
    for row in rankings:
        if len(row["results"]) != 4:
            raise ValueError("Missing diagnostic method")
        for ranking in row["results"].values():
            if len(ranking) != 50 or len(set(ranking)) != 50 or not set(ranking) <= clause_ids:
                raise ValueError("Invalid diagnostic clause ranking")
    print(f"PASS: {links} local links; {len(manifest['preserved_files'])} relocated artifact hashes; "
          f"{run_count} saved baseline runs with matching metrics/evidence; {len(inputs)} diagnostic inputs.")


if __name__ == "__main__":
    main()
