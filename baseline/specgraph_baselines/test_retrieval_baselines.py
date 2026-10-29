"""Regression checks for real input formats, clause budgets and evaluation."""
import argparse
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import retrieval_baselines as baseline


def chunk(chunk_id, clause_id, text="text"):
    return {"chunk_id": chunk_id, "clause_id": clause_id, "title": "Title",
            "text": text, "spec_edition": "N4950"}


def query(gold=None):
    return {"query_id": "q1", "query": "increment", "code": "i++",
            "gold_clause_ids": ["a"] if gold is None else gold}


class RetrievalTests(unittest.TestCase):
    def test_actual_object_stream_and_graph(self):
        root = baseline.PROJECT_ROOT
        chunks = [baseline.normalize_chunk(row) for row in baseline.read_records(root / "data/processed/chunks.jsonl")]
        queries = baseline.read_records(root / "data/benchmark/spec_benchmark_v1.jsonl")
        graph = json.loads((root / "data/graph/spec_graph.json").read_text(encoding="utf-8"))
        report = baseline.validate_data(chunks, queries, graph)
        self.assertGreater(report["chunks"], report["clauses"])
        self.assertEqual(report["labeled_queries"], len(queries))
        self.assertEqual(report["graph"]["nodes"], len(graph["nodes"]))

    def test_input_formats_and_errors(self):
        rows = [{"a": 1}, {"b": "line\nbreak"}]
        with tempfile.TemporaryDirectory() as location:
            path = Path(location) / "data.jsonl"
            for text in (json.dumps(rows), "\n".join(map(json.dumps, rows)),
                         "\ufeff" + "\n\n".join(json.dumps(row, indent=2) for row in rows)):
                path.write_text(text, encoding="utf-8")
                self.assertEqual(baseline.read_records(path), rows)
            path.write_text('{"a": 1}\n{"broken":', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data.jsonl:2"):
                baseline.read_records(path)
            path.write_text('[1, 2]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON objects"):
                baseline.read_records(path)

    def test_legacy_clause_record_still_supported(self):
        row = baseline.normalize_chunk({"clause_id": "a", "title": "A", "text": "Text"})
        self.assertEqual(row["chunk_id"], "a")
        baseline.validate_data([row], [query()])

    def test_candidate_budget_counts_clauses_not_chunks(self):
        chunks = [chunk("a1", "a"), chunk("a2", "a"), chunk("a3", "a"), chunk("b1", "b")]
        ranking, evidence = baseline.collapse_to_clauses([c["chunk_id"] for c in chunks],
                                                       {c["chunk_id"]: c for c in chunks}, 2)
        self.assertEqual(ranking, ["a", "b"])
        self.assertEqual(evidence, ["a1", "b1"])
        self.assertEqual(baseline.metrics_at_k(ranking, ["a", "b"], 2)["recall@2"], 1)

    def test_hand_calculated_metrics_and_duplicate_guard(self):
        result = baseline.metrics_at_k(["a", "b", "c"], ["a", "c"], 3)
        self.assertEqual(result["recall@3"], 1)
        self.assertEqual(result["precision@3"], 2 / 3)
        self.assertEqual(result["complete_coverage@3"], 1)
        self.assertAlmostEqual(result["ndcg@3"], 1.5 / (1 + 1 / math.log2(3)))
        self.assertEqual(baseline.metrics_at_k(["a"], ["a"], 5)["precision@5"], 0.2)
        with self.assertRaisesRegex(ValueError, "distinct"):
            baseline.metrics_at_k(["a", "a"], ["a"], 2)
        with self.assertRaises(ValueError):
            baseline.metrics_at_k(["a"], [], 2)

    def test_macro_aggregation_does_not_weight_by_gold_count(self):
        rows = [{"difficulty": "single", "gold_clause_ids": ["a"], "results": {"m": ["a"]}},
                {"difficulty": "multi", "gold_clause_ids": ["a", "b", "c"], "results": {"m": ["b"]}},
                {"difficulty": "unlabeled", "gold_clause_ids": [], "results": {"m": ["b"]}}]
        result = baseline.aggregate(rows, ["m"], [1])
        self.assertEqual(result["evaluated_queries"], 2)
        self.assertAlmostEqual(result["overall"]["m"]["recall@1"], 0.66667)
        self.assertEqual(result["by_gold_count"]["3"]["n"], 1)

    def test_annotations_are_not_retrieval_input(self):
        q = {**query(), "concept": "SECRET", "expected_property": "SECRET",
             "source": "SECRET", "difficulty": "SECRET"}
        self.assertEqual(baseline.query_text(q), "increment\ni++")
        c = {**chunk("a1", "a"), "cross_references": ["SECRET"], "source_url": "SECRET"}
        self.assertEqual(baseline.document_text(c), "Title\ntext")

    def test_cpp_operator_tokenization(self):
        self.assertEqual(baseline.tokenize("std::move(x) <<= 2; ++i && j != 0"),
                         ["std::move", "(", "x", ")", "<<=", "2", ";", "++", "i", "&&", "j", "!=", "0"])

    def test_content_preprocessing_keeps_operators_without_annotation_hints(self):
        text = "What is the behavior of a++ && b++, with bool?"
        tokens = baseline.lexical_tokens(text, "content")
        self.assertEqual(tokens, ["behavior", "++", "&&", "++", ",", "bool"])
        self.assertEqual(baseline.lexical_tokens(text, "raw"), baseline.tokenize(text))

    def test_bm25_hand_computation_and_tie_break(self):
        bm25 = baseline.BM25([["x", "x"], ["y", "y"]])
        self.assertAlmostEqual(bm25.get_scores(["x"])[0], math.log(2) * 5 / 3.5)
        self.assertEqual(bm25.get_scores(["x"])[1], 0)
        chunks = [chunk("z", "z"), chunk("a", "a")]
        self.assertEqual(baseline.rank_bm25("unseen", bm25, chunks, 2), ["a", "z"])

    def test_rrf_missing_items_duplicates_and_ties(self):
        self.assertEqual(baseline.reciprocal_rank_fusion([["a", "b"], ["b", "a"]], 2), ["a", "b"])
        self.assertEqual(baseline.reciprocal_rank_fusion([["a", "a", "b"], ["b"]], 2), ["b", "a"])

    def test_weighted_rrf_and_invalid_weights(self):
        ranks = [["a", "b"], ["b", "a"]]
        self.assertEqual(baseline.reciprocal_rank_fusion(ranks, 2, weights=[1, 2]), ["b", "a"])
        self.assertEqual(baseline.reciprocal_rank_fusion(ranks, 2, weights=[1, 1]),
                         baseline.reciprocal_rank_fusion(ranks, 2))
        self.assertEqual(baseline.reciprocal_rank_fusion(ranks, 2, weights=[0, 1]), ["b", "a"])
        for weights in ([1], [0, 0], [-1, 1], [1, float("nan")], [1, float("inf")]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                baseline.reciprocal_rank_fusion(ranks, 2, weights=weights)

    def test_question_only_excludes_code_and_annotations(self):
        q = {**query(), "concept": "SECRET", "gold_clause_ids": ["SECRET"]}
        self.assertEqual(baseline.query_text(q, "question"), "increment")
        self.assertEqual(baseline.query_text(q), "increment\ni++")
        with self.assertRaises(ValueError):
            baseline.query_text(q, "concept")

    def test_reject_invalid_ids_labels_and_editions(self):
        corpus = [chunk("a1", "a"), chunk("a2", "a")]
        baseline.validate_data(corpus, [query()])  # Multiple chunks per clause are valid.
        invalid = [([corpus[0], corpus[0]], [query()]),
                   (corpus, [query(), query()]),
                   (corpus, [query(["missing"])]),
                   (corpus, [query("a")]),
                   (corpus, [query(["a", "a"])]),
                   (corpus, [{**query(), "code": 42}]),
                   (corpus, [{**query(), "spec_edition": "N3337"}]),
                   ([], [query()]), (corpus, [])]
        for chunks, queries in invalid:
            with self.subTest(chunks=chunks, queries=queries), self.assertRaises(ValueError):
                baseline.validate_data(chunks, queries)

    def test_graph_null_chunk_is_valid_but_bad_links_fail(self):
        corpus = [chunk("a1", "a")]
        graph = {"nodes": [{"node_id": "clause:a", "clause_id": "a", "node_type": "clause", "chunk_id": None}], "edges": []}
        baseline.validate_data(corpus, [query()], graph)
        bad = copy.deepcopy(graph)
        bad["nodes"][0]["chunk_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "invalid chunk_id"):
            baseline.validate_data(corpus, [query()], bad)
        bad = copy.deepcopy(graph)
        bad["edges"].append({"source": "clause:a", "target": "missing"})
        with self.assertRaisesRegex(ValueError, "missing endpoint"):
            baseline.validate_data(corpus, [query()], bad)

    def test_graph_audit_normalizes_definition_and_syntax_anchors(self):
        corpus = [chunk("a1", "a"), chunk("b1", "b")]
        corpus[0]["cross_references"] = [{"target_clause_id": "b#nt:expression"}]
        graph = {"nodes": [{"node_id": f"clause:{cid}", "clause_id": cid, "node_type": "clause"}
                           for cid in "ab"], "edges": []}
        report = baseline.validate_data(corpus, [query()], graph)
        self.assertEqual(report["graph"]["normalized_reference_edges_missing"],
                         [{"source": "clause:a", "target": "clause:b"}])


class Matrix(list):
    """Minimal deterministic embedding fixture; no ML installation needed."""
    def __matmul__(self, vector):
        return [sum(a * b for a, b in zip(row, vector)) for row in self]


class PipelineTests(unittest.TestCase):
    def test_dense_hybrid_rerank_preserve_distinct_candidates_and_evidence(self):
        seen = {}

        class Encoder:
            max_seq_length = 512

            def __init__(self, name):
                pass

            def tokenizer(self, text, truncation=False):
                return {"input_ids": text.split()}

            def encode_document(self, texts, **kwargs):
                # Two strongest chunks both belong to clause a.
                seen["documents"] = texts
                return Matrix([[1, 0], [0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])

            def encode_query(self, texts, **kwargs):
                seen["queries"] = texts
                seen["prefix"] = kwargs.get("prompt")
                return [[1, 0]]

        class Reranker:
            def __init__(self, name):
                pass

            def predict(self, pairs, **kwargs):
                seen["rerank_pairs"] = pairs
                return list(range(len(pairs)))

        with tempfile.TemporaryDirectory() as location:
            root = Path(location)
            chunks = [chunk("a1", "a", "increment"), chunk("a2", "a", "increment increment"),
                      chunk("b1", "b", "increment"), chunk("c1", "c", "unrelated")]
            (root / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
            (root / "queries.json").write_text(json.dumps([query(["a", "b"])]), encoding="utf-8")
            graph = {"nodes": [{"node_id": f"node:{cid}", "clause_id": cid, "node_type": "clause"} for cid in "abc"], "edges": []}
            (root / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
            args = argparse.Namespace(chunks=str(root / "chunks.json"), queries=str(root / "queries.json"),
                                      graph=str(root / "graph.json"), output_dir=str(root / "out"),
                                      methods=["bm25", "dense", "hybrid_rrf"], rerank=True,
                                      embedding_model=baseline.DEFAULT_EMBEDDING, query_prefix=None,
                                      dense_backend="exact", chroma_dir=str(root / "chroma"),
                                      reranker_model="fixture", candidate_k=2, top_k=[1, 2], rrf_k=60, dry_run=False)
            module = types.SimpleNamespace(SentenceTransformer=Encoder, CrossEncoder=Reranker)
            with patch.dict(sys.modules, {"sentence_transformers": module}), contextlib.redirect_stdout(io.StringIO()):
                baseline.run(args)
            row = baseline.read_records(root / "out/ranked_results.jsonl")[0]
            self.assertEqual(row["results"]["dense"], ["a", "b"])
            self.assertEqual(row["evidence_chunk_ids"]["dense"], ["a1", "b1"])
            self.assertEqual(set(row["results"]["hybrid_rrf"]), set(row["results"]["hybrid_cross_encoder"]))
            self.assertEqual(len(seen["rerank_pairs"]), 2)
            self.assertEqual(seen["queries"], ["increment\ni++"])
            self.assertTrue(seen["prefix"].startswith("Represent this sentence"))
            for method, clauses in row["results"].items():
                self.assertEqual(row["node_results"][method], [f"node:{cid}" for cid in clauses])
                by_id = {c["chunk_id"]: c for c in chunks}
                for cid, eid in zip(clauses, row["evidence_chunk_ids"][method]):
                    self.assertEqual(cid, by_id[eid]["clause_id"])

    def test_actual_bm25_cli_requires_no_models(self):
        with tempfile.TemporaryDirectory() as location:
            result = subprocess.run([sys.executable, str(Path(baseline.__file__)), "--methods", "bm25",
                                     "--graph", str(baseline.PROJECT_ROOT / "data/graph/spec_graph.json"),
                                     "--output-dir", location], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = baseline.read_records(Path(location) / "ranked_results.jsonl")
            self.assertGreater(len(rows), 0)
            for row in rows:
                self.assertEqual(list(row["results"]), ["bm25"])
                self.assertEqual(len(row["results"]["bm25"]), 50)
                self.assertEqual(len(set(row["results"]["bm25"])), 50)
                self.assertEqual(len(row["evidence_chunk_ids"]["bm25"]), 50)
            metrics = json.loads((Path(location) / "metrics.json").read_text())
            self.assertEqual(metrics["evaluated_queries"], len(rows))


if __name__ == "__main__":
    unittest.main()
