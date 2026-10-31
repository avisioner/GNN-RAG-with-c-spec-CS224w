"""Full clause-score vectors for every benchmark query, from Erica's retrievers.

Her CLI saves only top-50 clause rankings (no scores), but graph expansion needs
a score for all 82 clauses. This wrapper reuses her functions and settings
unchanged (document_text, query_text, BM25, lexical_tokens, BGE query prefix)
and records the raw scores. Following her collapse rule (a clause's rank is
that of its best chunk), a clause's score is the max over its chunks.

    python graph/specgraph_graph/clause_scores.py     # writes artifacts/scores/question_scores.json

The cached file lets run_milestone.py run with numpy only (e.g. in Colab
without downloading the model).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from common import BASELINE_DIR, SCORES_DIR, corpus_clauses, erica, load_benchmark, load_chunks, write_json

QUERY_MODE = "question"      # Erica's reported reference input mode
LEXICAL_MODE = "content"     # her reported BM25 preprocessing
SCORES_PATH = SCORES_DIR / "question_scores.json"


def chunk_to_clause_max(chunk_scores: np.ndarray, chunk_clause: list[str], clauses: list[str]) -> np.ndarray:
    """Max chunk score per clause, in `clauses` order."""
    index = {c: i for i, c in enumerate(clauses)}
    out = np.full(len(clauses), -np.inf)
    for s, c in zip(chunk_scores, chunk_clause):
        out[index[c]] = max(out[index[c]], float(s))
    return out


def compute_scores(model_name: str = erica.DEFAULT_EMBEDDING, threads: int = 4) -> dict[str, Any]:
    chunks = [erica.normalize_chunk(r) for r in load_chunks()]
    queries = load_benchmark()
    clauses = corpus_clauses(chunks)
    chunk_clause = [c["clause_id"] for c in chunks]
    texts = [erica.document_text(c) for c in chunks]
    qtexts = [erica.query_text(q, QUERY_MODE) for q in queries]

    bm25 = erica.BM25([erica.lexical_tokens(t, LEXICAL_MODE) for t in texts])
    bm25_scores = [chunk_to_clause_max(np.array(bm25.get_scores(erica.lexical_tokens(t, LEXICAL_MODE))),
                                       chunk_clause, clauses) for t in qtexts]

    os.environ.setdefault("HF_HOME", str(BASELINE_DIR.parent / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import torch
    torch.set_num_threads(threads)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    prefix = ("Represent this sentence for searching relevant passages: "
              if model_name.startswith("BAAI/bge-") else getattr(model, "prompts", {}).get("query", ""))
    doc_vecs = model.encode_document(texts, normalize_embeddings=True, show_progress_bar=False)
    q_vecs = model.encode_query(qtexts, prompt=prefix, normalize_embeddings=True, show_progress_bar=False)
    dense_scores = [chunk_to_clause_max(doc_vecs @ v, chunk_clause, clauses) for v in q_vecs]

    from importlib.metadata import version
    return {
        "clause_ids": clauses, "query_ids": [q["query_id"] for q in queries],
        "query_mode": QUERY_MODE, "lexical_mode": LEXICAL_MODE,
        "embedding_model": model_name, "query_prefix": prefix, "clause_score": "max over chunks",
        "packages": {p: version(p) for p in ("numpy", "torch", "sentence-transformers", "transformers")},
        "dense": [[round(float(x), 7) for x in row] for row in dense_scores],
        "bm25": [[round(float(x), 6) for x in row] for row in bm25_scores],
    }


def load_scores(path: Path = SCORES_PATH) -> dict[str, Any]:
    """Cached scores with `dense` / `bm25` as numpy arrays of shape [n_queries, n_clauses]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("dense", "bm25"):
        data[key] = np.array(data[key])
    return data


if __name__ == "__main__":
    write_json(SCORES_PATH, compute_scores())
    print(f"Wrote {SCORES_PATH}")
