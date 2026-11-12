"""Paths and imports shared by the GNN milestone module.

Everything here runs on the *old* 82-clause corpus (commit 7a29ead), extracted to
artifacts/corpus_v1/chunks.jsonl, because the benchmark queries were written
against it. data/processed/chunks.jsonl now holds Monami's 460-clause parse and
is not used.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

GNN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GNN_DIR.parents[1]
GRAPH_MODULE_DIR = PROJECT_ROOT / "graph" / "specgraph_graph"
if str(GRAPH_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_MODULE_DIR))

import common as graph_common  # noqa: E402  (Aaron's Oct 30 module; imports Erica's module too)
from common import erica, load_benchmark, load_chunks, corpus_clauses, write_json  # noqa: E402,F401

CORPUS_V1_PATH = PROJECT_ROOT / "artifacts" / "corpus_v1" / "chunks.jsonl"
CORPUS_V1_COMMIT = "7a29ead"
CORPUS_V1_SHA256 = "8dde85cbed9ab15256a8bf10b9bd38946dfbf5f7ed0e46a56ef0de44c7ca5c35"
OUT_DIR = PROJECT_ROOT / "results" / "gnn_milestone"
CACHE_DIR = OUT_DIR / "cache"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_corpus_v1() -> list[dict[str, Any]]:
    """Old-corpus chunks via Erica's reader, after checking the file's bytes."""
    if sha256(CORPUS_V1_PATH) != CORPUS_V1_SHA256:
        raise ValueError(f"{CORPUS_V1_PATH} is not the 7a29ead corpus. Recreate it with: "
                         f"git show {CORPUS_V1_COMMIT}:data/processed/chunks.jsonl > {CORPUS_V1_PATH}")
    return [erica.normalize_chunk(r) for r in load_chunks(CORPUS_V1_PATH)]


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]
