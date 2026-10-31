"""Shared paths, loaders and ID normalization for the graph milestone.

Erica's retrieval module is imported, not copied, so the corpus reader and the
metric definitions stay identical to the text baselines.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "baseline" / "specgraph_baselines"
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "spec_benchmark_v1.jsonl"
SPEC_GRAPH_PATH = PROJECT_ROOT / "data" / "graph" / "spec_graph.json"
CLAUSE_GRAPH_PATH = PROJECT_ROOT / "artifacts" / "graph" / "clause_graph.json"
SCORES_DIR = PROJECT_ROOT / "artifacts" / "scores"
RESULTS_DIR = PROJECT_ROOT / "results" / "milestone"

if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import retrieval_baselines as erica  # noqa: E402  (needs the sys.path entry above)

DIFFICULTIES = ("single_clause", "two_clause", "multi_hop")
_DIFFICULTY_ALIASES = {
    "single": "single_clause", "single_clause": "single_clause", "single-clause": "single_clause",
    "single clause": "single_clause",
    "two": "two_clause", "two_clause": "two_clause", "two-clause": "two_clause", "two clause": "two_clause",
    "multi": "multi_hop", "multi_hop": "multi_hop", "multi-hop": "multi_hop", "multi hop": "multi_hop",
    "multi_clause": "multi_hop", "multi-clause": "multi_hop", "multi clause": "multi_hop",
}


def norm_id(raw: str) -> str:
    """Map any chunk, node or anchor ID to its bare clause ID.

    >>> norm_id("clause:expr.pre#p2")
    'expr.pre'
    >>> norm_id("dcl.init.general#nt:initializer")
    'dcl.init.general'
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Expected a nonempty ID string, got {raw!r}")
    s = raw.strip()
    s = s.removeprefix("clause:")
    return s.split("#", 1)[0]


def normalize_difficulty(raw: str) -> str:
    """Normalize a difficulty label to one of DIFFICULTIES; raise on unknown labels."""
    key = " ".join(raw.strip().lower().replace("_", " ").replace("-", " ").split())
    if key not in _DIFFICULTY_ALIASES:
        raise ValueError(f"Unknown difficulty label: {raw!r}")
    return _DIFFICULTY_ALIASES[key]


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict[str, Any]]:
    """Read chunks with Erica's reader (handles the pretty-printed multi-line JSONL)."""
    return erica.read_records(path)


def load_benchmark(path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    """Read benchmark queries, adding normalized `gold` and `difficulty_norm` fields.

    The raw fields are left untouched so the validation report can show them.
    """
    queries = erica.read_records(path)
    for q in queries:
        q["gold"] = list(dict.fromkeys(norm_id(g) for g in q.get("gold_clause_ids", [])))
        q["difficulty_norm"] = normalize_difficulty(q.get("difficulty", ""))
    return queries


def load_spec_graph(path: Path = SPEC_GRAPH_PATH) -> dict[str, Any]:
    """Read the shared spec_graph.json as-is."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def corpus_clauses(chunks: list[dict[str, Any]]) -> list[str]:
    """Sorted distinct clause IDs; this order indexes every score vector."""
    return sorted({norm_id(c["clause_id"]) for c in chunks})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
