"""BM25 and dense clause scores for arbitrary query text, with Erica's functions.

Same settings as graph/specgraph_graph/clause_scores.py (which produced the saved
benchmark scores): `document_text` for chunks, `lexical_tokens(·, "content")`
for BM25, BGE-small with the BGE query prefix, normalized embeddings. This adds
the mean-over-chunks dense score, which needs the per-chunk matrix.

`check_benchmark_reproduction()` re-scores the 49 benchmark questions this way
and requires an exact match with artifacts/scores/question_scores.json.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from gnn_common import CACHE_DIR, GRAPH_MODULE_DIR, config_hash, corpus_clauses, erica, load_benchmark, load_corpus_v1

import clause_scores  # noqa: E402  (Aaron's Oct 30 module, on sys.path via gnn_common)

LEXICAL_MODE = clause_scores.LEXICAL_MODE
MODEL_NAME = erica.DEFAULT_EMBEDDING
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


class Scorer:
    """Holds the BM25 index and chunk embeddings; scores lists of query strings."""

    def __init__(self, threads: int = 4) -> None:
        self.chunks = load_corpus_v1()
        self.clauses = corpus_clauses(self.chunks)
        self.chunk_clause = [c["clause_id"] for c in self.chunks]
        idx = {c: i for i, c in enumerate(self.clauses)}
        self.member = np.zeros((len(self.clauses), len(self.chunks)))     # clause x chunk indicator
        for j, c in enumerate(self.chunk_clause):
            self.member[idx[c], j] = 1.0
        texts = [erica.document_text(c) for c in self.chunks]
        self.bm25 = erica.BM25([erica.lexical_tokens(t, LEXICAL_MODE) for t in texts])

        os.environ.setdefault("HF_HOME", str(GRAPH_MODULE_DIR.parents[1] / "baseline" / ".cache" / "huggingface"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        import torch
        torch.set_num_threads(threads)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME)
        self.doc_vecs = self.model.encode_document(texts, normalize_embeddings=True, show_progress_bar=False)

    def score(self, qtexts: list[str]) -> dict[str, np.ndarray]:
        """[n_queries, n_clauses] arrays: dense_max, dense_mean, bm25_max."""
        q_vecs = self.model.encode_query(qtexts, prompt=BGE_PREFIX, normalize_embeddings=True,
                                         show_progress_bar=False)
        chunk_dense = q_vecs @ self.doc_vecs.T
        dense_max = np.stack([clause_scores.chunk_to_clause_max(r, self.chunk_clause, self.clauses)
                              for r in chunk_dense])
        dense_mean = chunk_dense @ self.member.T / self.member.sum(axis=1)
        bm25_max = np.stack([clause_scores.chunk_to_clause_max(
            np.array(self.bm25.get_scores(erica.lexical_tokens(t, LEXICAL_MODE))), self.chunk_clause, self.clauses)
            for t in qtexts])
        return {"dense_max": dense_max, "dense_mean": dense_mean, "bm25_max": bm25_max}


_SCORER: Scorer | None = None


def get_scorer() -> Scorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer()
    return _SCORER


def cached_scores(qtexts: list[str], tag: str) -> dict[str, np.ndarray]:
    """Score `qtexts`, caching the arrays under results/gnn_milestone/cache/."""
    key = config_hash({"tag": tag, "texts": qtexts, "model": MODEL_NAME, "lexical": LEXICAL_MODE,
                       "prefix": BGE_PREFIX, "v": 1})
    path = CACHE_DIR / f"scores_{tag}_{key}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    out = get_scorer().score(qtexts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def benchmark_scores() -> dict[str, Any]:
    """Question-only scores for the 49 benchmark queries, checked against the saved file."""
    queries = load_benchmark()
    texts = [erica.query_text(q, clause_scores.QUERY_MODE) for q in queries]
    out = cached_scores(texts, "benchmark")
    saved = clause_scores.load_scores()
    if saved["query_ids"] != [q["query_id"] for q in queries]:
        raise ValueError("Saved question_scores.json is in a different query order")
    # Saved file is rounded to 7 (dense) / 6 (bm25) decimals.
    if not (np.abs(out["dense_max"] - saved["dense"]).max() < 1e-6
            and np.abs(out["bm25_max"] - saved["bm25"]).max() < 1e-5):
        raise ValueError("Re-scored benchmark does not match artifacts/scores/question_scores.json")
    return out


if __name__ == "__main__":
    s = benchmark_scores()
    print("benchmark re-scoring matches question_scores.json:", {k: v.shape for k, v in s.items()})
