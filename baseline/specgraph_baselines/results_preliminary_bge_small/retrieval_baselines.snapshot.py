#!/usr/bin/env python3
"""SpecGraph-RAG: retrieve chunks, evaluate distinct clauses.

BM25 uses only the standard library. Dense retrieval, clause-level RRF and
optional cross-encoder reranking use sentence-transformers; Chroma is optional.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import re
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*(?:::[A-Za-z_][A-Za-z_0-9]*)*"
    r"|<<=|>>=|->\*|<=>|\+\+|--|->|&&|\|\||==|!=|<=|>=|<<|>>"
    r"|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\d+|[^\s]"
)


def read_records(filename: str | Path) -> list[dict[str, Any]]:
    """Accept JSON arrays, JSONL and multiline JSON object streams.

    The supplied chunks.jsonl contains pretty-printed objects spanning lines.
    """
    path = Path(filename)
    text = path.read_text(encoding="utf-8-sig")
    decoder, records, position = json.JSONDecoder(), [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        try:
            obj, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc
        records.extend(obj if isinstance(obj, list) else [obj])
    if not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected JSON objects in {path}")
    return records


def normalize_chunk(row: dict[str, Any]) -> dict[str, Any]:
    clause_id = row.get("clause_id") or row.get("id")
    chunk_id = row.get("chunk_id") or clause_id
    body = row.get("text") or row.get("content") or row.get("body")
    for name, value in (("clause_id", clause_id), ("chunk_id", chunk_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Every chunk must have a nonempty {name}")
    if not isinstance(body, str) or not body.strip():
        raise ValueError(f"Chunk {chunk_id}: missing text/content/body")
    title = row.get("title", "")
    if not isinstance(title, str):
        raise ValueError(f"Chunk {chunk_id}: title must be a string")
    return {**row, "chunk_id": chunk_id, "clause_id": clause_id,
            "title": title, "text": body}


def document_text(chunk: dict[str, Any]) -> str:
    # Provenance, cross-reference metadata and annotations are never indexed.
    return f"{chunk['title']}\n{chunk['text']}".strip()


def query_text(query: dict[str, Any]) -> str:
    return "\n".join(query[key] for key in ("query", "code") if query.get(key)).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


STOPWORDS = frozenset("a an the and or of to in on for from by with as at is are was were be been being this that these those it its if then else what which when where whether how does do did may can could should would will shall has have had not both each any all same following".split())


def lexical_tokens(text: str, mode: str = "raw") -> list[str]:
    tokens = tokenize(text)
    if mode == "raw":
        return tokens
    # General preprocessing; no query annotations or clause-specific hints.
    # Preserve operators, including comma. Suppress single-letter variables,
    # prose stopwords and structural delimiters that swamp lexical matching.
    return [t for t in tokens if t not in STOPWORDS and t not in {"(", ")", "[", "]", "{", "}", ";", ":", ".", "?"}
            and not (len(t) == 1 and t.isalpha())]


class BM25:
    """Okapi BM25 with positive IDF and preserved query term multiplicity.

    IDF(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5)). This differs
    explicitly from rank_bm25's negative-IDF flooring variant.
    """
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        if not documents or not any(documents):
            raise ValueError("BM25 requires a nonempty tokenized corpus")
        self.k1, self.b = k1, b
        self.counts = [Counter(doc) for doc in documents]
        self.lengths = [len(doc) for doc in documents]
        self.average_length = mean(self.lengths)
        df = Counter(term for counts in self.counts for term in counts)
        self.idf = {term: math.log1p((len(documents) - n + 0.5) / (n + 0.5))
                    for term, n in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        query_counts = Counter(query)
        scores = []
        for counts, length in zip(self.counts, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.average_length)
            scores.append(sum(
                n * self.idf.get(term, 0.0) * counts[term] * (self.k1 + 1)
                / (counts[term] + norm)
                for term, n in query_counts.items() if counts[term]
            ))
        return scores


def rank_bm25(query: str, bm25: Any, chunks: list[dict[str, Any]], limit: int,
              lexical_mode: str = "raw") -> list[str]:
    scores = bm25.get_scores(lexical_tokens(query, lexical_mode))
    ranks = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"]))
    return [chunks[i]["chunk_id"] for i in ranks[:limit]]


def collapse_to_clauses(ranking: list[str], chunks_by_id: dict[str, dict[str, Any]],
                        limit: int) -> tuple[list[str], list[str]]:
    """Best chunk wins; candidate and evaluation budgets count clauses."""
    if limit < 1:
        raise ValueError("Clause limit must be positive")
    clauses, evidence, seen = [], [], set()
    for chunk_id in ranking:
        clause_id = chunks_by_id[chunk_id]["clause_id"]
        if clause_id not in seen:
            seen.add(clause_id)
            clauses.append(clause_id)
            evidence.append(chunk_id)
            if len(clauses) == limit:
                break
    return clauses, evidence


def reciprocal_rank_fusion(rankings: list[list[str]], limit: int, rrf_k: int = 60) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(dict.fromkeys(ranking), start=1):
            score[item_id] += 1.0 / (rrf_k + rank)
    return sorted(score, key=lambda item_id: (-score[item_id], item_id))[:limit]


def metrics_at_k(retrieved: list[str], gold: list[str], k: int) -> dict[str, float]:
    if not gold or k < 1:
        raise ValueError("Metrics require nonempty gold and positive k")
    if len(set(retrieved)) != len(retrieved):
        raise ValueError("Rankings must contain distinct clause IDs")
    retrieved_k, gold_set = retrieved[:k], set(gold)
    hits = gold_set.intersection(retrieved_k)
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(retrieved_k) if cid in gold_set)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold_set))))
    return {f"recall@{k}": len(hits) / len(gold_set),
            f"precision@{k}": len(hits) / k,
            f"complete_coverage@{k}": float(gold_set.issubset(retrieved_k)),
            f"ndcg@{k}": dcg / ideal}


def aggregate(records: list[dict[str, Any]], methods: list[str], ks: list[int]) -> dict[str, Any]:
    labeled = [row for row in records if row["gold_clause_ids"]]
    output: dict[str, Any] = {"evaluated_queries": len(labeled), "overall": {},
                              "by_difficulty": {}, "by_gold_count": {}}

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        table = {}
        for method in methods:
            per_query = [{key: value for k in ks for key, value in
                          metrics_at_k(row["results"][method], row["gold_clause_ids"], k).items()}
                         for row in subset]
            table[method] = {name: round(mean(r[name] for r in per_query), 5)
                             for name in per_query[0]}
        return table

    if labeled:
        output["overall"] = summarize(labeled)
        for field, group_key in (("difficulty", "by_difficulty"), ("gold_count", "by_gold_count")):
            groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for row in labeled:
                key = row["difficulty"] if field == "difficulty" else len(set(row["gold_clause_ids"]))
                groups[key].append(row)
            for key, subset in sorted(groups.items()):
                output[group_key][str(key)] = {"n": len(subset), "metrics": summarize(subset)}
    return output


def validate_data(chunks: list[dict[str, Any]], queries: list[dict[str, Any]],
                  graph: dict[str, Any] | None = None) -> dict[str, Any]:
    if not chunks or not queries:
        raise ValueError("Corpus and queries must both be nonempty")
    chunk_ids = [c["chunk_id"] for c in chunks]
    clause_ids = {c["clause_id"] for c in chunks}
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Duplicate chunk IDs in corpus")
    query_ids = set()
    for q in queries:
        qid = q.get("query_id")
        if not isinstance(qid, str) or not qid.strip() or qid in query_ids:
            raise ValueError(f"Missing or duplicate query_id: {qid!r}")
        query_ids.add(qid)
        for key in ("query", "code"):
            if key in q and not isinstance(q[key], str):
                raise ValueError(f"Query {qid}: {key} must be a string")
        if not query_text(q):
            raise ValueError(f"Query {qid}: query/code is empty")
        gold = q.get("gold_clause_ids", [])
        if not isinstance(gold, list) or any(not isinstance(cid, str) or not cid for cid in gold):
            raise ValueError(f"Query {qid}: gold_clause_ids must be a list of strings")
        if len(set(gold)) != len(gold):
            raise ValueError(f"Query {qid}: duplicate gold clause IDs")
        unknown = set(gold) - clause_ids
        if unknown:
            raise ValueError(f"Query {qid}: gold clauses absent from corpus: {sorted(unknown)}")
        if not isinstance(q.get("difficulty", "unspecified"), str):
            raise ValueError(f"Query {qid}: difficulty must be a string")
    editions = {row["spec_edition"] for row in chunks + queries if row.get("spec_edition")}
    if graph and graph.get("spec_edition"):
        editions.add(graph["spec_edition"])
    if len(editions) > 1:
        raise ValueError(f"Mixed specification editions: {sorted(editions)}")
    refs = {r["target_clause_id"].split("#", 1)[0]
            for c in chunks for r in c.get("cross_references", [])}
    gold_counts = Counter(cid for q in queries for cid in q.get("gold_clause_ids", []))
    report: dict[str, Any] = {
        "chunks": len(chunks), "clauses": len(clause_ids), "queries": len(queries),
        "labeled_queries": sum(bool(q.get("gold_clause_ids")) for q in queries),
        "spec_editions": sorted(editions),
        "chunk_types": dict(Counter(c.get("chunk_type", "unspecified") for c in chunks)),
        "difficulty_counts": dict(Counter(q.get("difficulty", "unspecified") for q in queries)),
        "expected_property_counts": dict(Counter(q.get("expected_property", "unspecified") for q in queries)),
        "gold_clause_frequency": dict(gold_counts.most_common()), "gold_clause_count": len(gold_counts),
        "cross_reference_targets_outside_corpus": sorted(refs - clause_ids),
        "chunks_with_literal_newline_escape": [c["chunk_id"] for c in chunks if "\\n" in c["text"]],
    }
    if graph is not None:
        nodes, edges = graph["nodes"], graph["edges"]
        node_ids = {n["node_id"] for n in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError("Duplicate graph node IDs")
        clause_nodes = {n["clause_id"]: n for n in nodes if n["node_type"] == "clause"}
        if len(clause_nodes) != sum(n["node_type"] == "clause" for n in nodes):
            raise ValueError("Multiple graph clause nodes for the same clause")
        if clause_ids - clause_nodes.keys():
            raise ValueError(f"Corpus clauses missing graph nodes: {sorted(clause_ids - clause_nodes.keys())}")
        chunks_by_id = {c["chunk_id"]: c for c in chunks}
        for node in nodes:
            chunk_id = node.get("chunk_id")
            if chunk_id and (chunk_id not in chunks_by_id or
                             chunks_by_id[chunk_id]["clause_id"] != node["clause_id"]):
                raise ValueError(f"Graph node {node['node_id']}: invalid chunk_id {chunk_id}")
        for edge in edges:
            if edge["source"] not in node_ids or edge["target"] not in node_ids:
                raise ValueError(f"Graph edge has missing endpoint: {edge}")
        nodes_by_clause = {n["clause_id"]: n["node_id"] for n in nodes}
        normalized_refs = {
            (nodes_by_clause[c["clause_id"]], nodes_by_clause[target])
            for c in chunks for ref in c.get("cross_references", [])
            if (target := ref["target_clause_id"].split("#", 1)[0]) in nodes_by_clause
            and target != c["clause_id"]
        }
        reference_edges = {(e["source"], e["target"]) for e in edges if e["edge_type"] == "references"}
        report["graph"] = {"nodes": len(nodes), "edges": len(edges),
                           "node_types": dict(Counter(n["node_type"] for n in nodes)),
                           "edge_types": dict(Counter(e["edge_type"] for e in edges)),
                           "clause_nodes_without_text": sorted(clause_nodes.keys() - clause_ids),
                           "normalized_reference_edge_count": len(normalized_refs),
                           "normalized_reference_edges_missing": [
                               {"source": source, "target": target}
                               for source, target in sorted(normalized_refs - reference_edges)]}
    return report


def rank_dense_exact(vector: Any, vectors: Any, chunks: list[dict[str, Any]]) -> list[str]:
    # Normalized arrays: dot product equals cosine similarity.
    scores = vectors @ vector
    return [chunks[i]["chunk_id"] for i in sorted(
        range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"]))]


def build_chroma_store(chunks: list[dict[str, Any]], model: Any, location: str, model_name: str):
    import chromadb

    client = chromadb.PersistentClient(path=location)
    payload = [(c["chunk_id"], c["clause_id"], document_text(c)) for c in chunks]
    signature = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()
    metadata = {"model": model_name, "corpus_sha": signature,
                "max_seq_length": model.max_seq_length, "schema": "chunks-v2"}
    name = "cpp23_spec_chunks_v2"
    try:
        coll = client.get_collection(name=name)
    except chromadb.errors.NotFoundError:
        coll = None
    if coll is not None and (any((coll.metadata or {}).get(k) != v for k, v in metadata.items())
                             or coll.count() != len(chunks)):
        client.delete_collection(name=name)
        coll = None
    if coll is None:
        coll = client.create_collection(name=name, metadata=metadata,
                                        configuration={"hnsw": {"space": "cosine"}})
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            vectors = model.encode_document([document_text(c) for c in batch],
                                             normalize_embeddings=True, show_progress_bar=False)
            coll.upsert(ids=[c["chunk_id"] for c in batch], embeddings=vectors.tolist(),
                        documents=[document_text(c) for c in batch],
                        metadatas=[{"clause_id": c["clause_id"], "title": c["title"]} for c in batch])
    return coll


def truncation_report(model: Any, texts: list[str]) -> dict[str, Any]:
    lengths = [len(model.tokenizer(text, truncation=False)["input_ids"]) for text in texts]
    return {"max_seq_length": model.max_seq_length,
            "truncated_inputs": sum(n > model.max_seq_length for n in lengths),
            "longest_input_tokens": max(lengths, default=0)}


def file_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    chunks = [normalize_chunk(row) for row in read_records(args.chunks)]
    queries = read_records(args.queries)
    graph = json.loads(Path(args.graph).read_text(encoding="utf-8-sig")) if args.graph else None
    audit = validate_data(chunks, queries, graph)
    print(f"Validated: {len(chunks)} chunks, {audit['clauses']} clauses, {len(queries)} queries")
    if args.dry_run:
        print(json.dumps(audit, indent=2, ensure_ascii=True))
        return audit
    methods = list(dict.fromkeys(args.methods))
    if args.rerank and "hybrid_cross_encoder" not in methods:
        methods.append("hybrid_cross_encoder")
    need_hybrid = any(m in methods for m in ("hybrid_rrf", "hybrid_cross_encoder"))
    need_bm25 = "bm25" in methods or need_hybrid
    need_dense = "dense" in methods or need_hybrid
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    corpus_texts = [document_text(c) for c in chunks]
    lexical_mode = getattr(args, "lexical_mode", "raw")
    bm25 = BM25([lexical_tokens(text, lexical_mode) for text in corpus_texts]) if need_bm25 else None
    embedding, vectors, coll, reranker = None, None, None, None
    query_prefix = args.query_prefix
    if need_dense:
        # CPU oversubscription can make small transformer runs much slower.
        try:
            import torch
            torch.set_num_threads(getattr(args, "threads", 4))
        except ImportError:
            pass  # The regression fixture does not require PyTorch.
        from sentence_transformers import SentenceTransformer
        embedding = SentenceTransformer(args.embedding_model)
        if query_prefix is None:
            query_prefix = ("Represent this sentence for searching relevant passages: "
                            if args.embedding_model.startswith("BAAI/bge-")
                            else getattr(embedding, "prompts", {}).get("query", ""))
        audit["embedding_documents"] = truncation_report(embedding, corpus_texts)
        audit["embedding_queries"] = truncation_report(embedding, [query_prefix + query_text(q) for q in queries])
        if args.dense_backend == "exact":
            vectors = embedding.encode_document(corpus_texts, normalize_embeddings=True,
                                                 show_progress_bar=True)
        else:
            coll = build_chroma_store(chunks, embedding, args.chroma_dir, args.embedding_model)
        print(f"Embedding truncation: {audit['embedding_documents']}")
    if "hybrid_cross_encoder" in methods:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(args.reranker_model)
    candidate_k = min(args.candidate_k, audit["clauses"])
    node_by_clause = {n["clause_id"]: n["node_id"] for n in graph["nodes"]
                      if n["node_type"] == "clause"} if graph else None
    rows = []
    for query in queries:
        text = query_text(query)
        chunk_rankings, results, evidence = {}, {}, {}
        if bm25 is not None:
            chunk_rankings["bm25"] = rank_bm25(text, bm25, chunks, len(chunks), lexical_mode)
        if embedding is not None:
            vector = embedding.encode_query([text], prompt=query_prefix,
                                             normalize_embeddings=True, show_progress_bar=False)[0]
            if coll is None:
                chunk_rankings["dense"] = rank_dense_exact(vector, vectors, chunks)
            else:
                # Fetch all chunks before collapse: duplicate-clause chunks must
                # not consume the distinct-clause candidate budget.
                hits = coll.query(query_embeddings=[vector.tolist()], n_results=len(chunks),
                                  include=["distances"])
                chunk_rankings["dense"] = [cid for cid, _ in sorted(
                    zip(hits["ids"][0], hits["distances"][0]), key=lambda pair: (pair[1], pair[0]))]
        for method, ranking in chunk_rankings.items():
            results[method], evidence[method] = collapse_to_clauses(ranking, chunks_by_id, candidate_k)
        if need_hybrid:
            results["hybrid_rrf"] = reciprocal_rank_fusion(
                [results["bm25"], results["dense"]], candidate_k, args.rrf_k)
            # Chunk fusion selects evidence only; the ranking above fuses
            # distinct clauses, preventing duplicate chunks from voting twice.
            fused_chunks = reciprocal_rank_fusion(list(chunk_rankings.values()), len(chunks), args.rrf_k)
            all_clauses, all_evidence = collapse_to_clauses(fused_chunks, chunks_by_id, audit["clauses"])
            chosen = dict(zip(all_clauses, all_evidence))
            evidence["hybrid_rrf"] = [chosen[cid] for cid in results["hybrid_rrf"]]
        if reranker is not None:
            candidates = list(zip(results["hybrid_rrf"], evidence["hybrid_rrf"]))
            scores = reranker.predict([(text, document_text(chunks_by_id[chunk_id]))
                                       for _, chunk_id in candidates], show_progress_bar=False)
            order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i][0]))
            results["hybrid_cross_encoder"] = [candidates[i][0] for i in order]
            evidence["hybrid_cross_encoder"] = [candidates[i][1] for i in order]
        row = {"query_id": query["query_id"], "difficulty": query.get("difficulty", "unspecified"),
               "gold_clause_ids": query.get("gold_clause_ids", []),
               "results": {m: results[m] for m in methods},
               "evidence_chunk_ids": {m: evidence[m] for m in methods}}
        if node_by_clause is not None:
            row["node_results"] = {m: [node_by_clause[cid] for cid in results[m]] for m in methods}
        rows.append(row)
        print(f"Retrieved {len(rows)}/{len(queries)}: {query['query_id']}", flush=True)
    metrics = aggregate(rows, methods, sorted(set(args.top_k)))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ranked_results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "corpus_manifest.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            item = {key: chunk[key] for key in ("chunk_id", "clause_id", "title", "source_url", "spec_edition") if key in chunk}
            if node_by_clause is not None:
                item["node_id"] = node_by_clause[chunk["clause_id"]]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    packages = {}
    for name in ("numpy", "sentence-transformers", "chromadb", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    config = {**vars(args), "methods": methods, "effective_candidate_k": candidate_k,
              "query_prefix": query_prefix if need_dense else None,
              "evaluation_unit": "distinct_clause", "evidence_unit": "best_chunk",
              "bm25": {"idf": "log1p((N-df+0.5)/(df+0.5))", "k1": 1.5, "b": 0.75},
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "packages": packages,
              "input_sha256": {"chunks": file_sha(args.chunks), "queries": file_sha(args.queries),
                               "script": file_sha(__file__)}}
    if args.graph:
        config["input_sha256"]["graph"] = file_sha(args.graph)
    for name, obj in (("metrics.json", metrics), ("data_audit.json", audit), ("run_config.json", config)):
        (output / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))
    print(f"Saved to: {output.resolve()}")
    return metrics


def self_test() -> None:
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_retrieval_baselines.py")
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", "--clauses", dest="chunks", default=str(PROJECT_ROOT / "data/processed/chunks.jsonl"))
    parser.add_argument("--queries", default=str(PROJECT_ROOT / "data/benchmark/spec_benchmark_v1.jsonl"))
    parser.add_argument("--graph", help="Validate graph mappings and export node IDs; never used for ranking")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--methods", nargs="+", choices=["bm25", "dense", "hybrid_rrf", "hybrid_cross_encoder"],
                        default=["bm25", "dense", "hybrid_rrf"])
    parser.add_argument("--dense-backend", choices=["exact", "chroma"], default="exact")
    parser.add_argument("--chroma-dir", default=".chroma_specgraph")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--query-prefix", default=None, help="Override query instruction (empty string disables it)")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--rerank", action="store_true", help="Also output hybrid_cross_encoder")
    parser.add_argument("--lexical-mode", choices=["raw", "content"], default="raw",
                        help="Optional BM25 stopword/variable/delimiter filtering; dense inputs stay unchanged")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="Audit files without models or output files")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--top-k", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--candidate-k", type=int, default=50, help="Distinct clause candidates after chunk collapse")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.top_k) < 1 or args.candidate_k < max(args.top_k) or args.rrf_k < 1:
        parser.error("k and rrf-k must be positive; candidate-k must be >= every top-k")
    try:
        run(args)
    except ImportError as exc:
        parser.exit(1, f"Missing optional dependency: {exc}. Use --methods bm25 for lexical-only retrieval; "
                       "for dense install baseline/specgraph_baselines/requirements.txt.\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Input/run error: {exc}\n")


if __name__ == "__main__":
    main()
