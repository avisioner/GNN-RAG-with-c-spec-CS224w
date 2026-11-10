import json
from pathlib import Path
import unittest

from query_variants import CONTROLS, VARIANTS, build_variants, operator_gloss
from run_experiments import METHODS, METRICS, ROOT, analyze, baseline, features


class QueryDiagnosticTests(unittest.TestCase):
    def test_all_query_pairs_preserve_code_and_exclude_label_fields(self):
        rewrites = json.loads(Path(__file__).with_name("explicit_questions.json").read_text(encoding="utf-8"))
        queries = baseline.read_records(ROOT / "data/benchmark/spec_benchmark_v1.jsonl")
        self.assertEqual(set(rewrites), {q["query_id"] for q in queries})
        for q in queries:
            texts = build_variants(q["query"], q["code"], rewrites[q["query_id"]])
            self.assertEqual(tuple(texts), VARIANTS)
            self.assertEqual(texts["question_only"], q["query"])
            self.assertEqual(texts["question_code"], baseline.query_text(q))
            self.assertEqual(texts["code_only"], q["code"])
            for variant, text in texts.items():
                if variant not in ("question_only", "question_domain"):
                    self.assertIn(q["code"], text)
                self.assertNotIn("gold_clause_ids", text)
                self.assertNotIn("expected_property", text)
                for cid in q["gold_clause_ids"]:
                    self.assertNotIn(cid, text)

    def test_field_order_changes_only_order(self):
        texts = build_variants("Can this run?", "f();", "May f() execute?")
        self.assertEqual(texts["labeled_code"].split("\n"),
                         [texts["code_first"].split("\n")[-1]] + texts["code_first"].split("\n")[:-1])
        self.assertIn("May f() execute?", texts["verbose_explicit"])

    def test_conditional_question_mark_is_not_a_second_question(self):
        texts = build_variants("What happens?", "return a ? b : c;", "What happens in a ? b : c?")
        self.assertIn("a ? b : c", texts["explicit_one_sentence"])

    def test_gloss_avoids_assignment_from_equality_and_compound_tokens(self):
        self.assertNotIn("assignment", operator_gloss("a == b; a != b; a <= b;"))
        self.assertIn("addition assignment", operator_gloss("a += 1;"))
        self.assertNotIn("=: assignment", operator_gloss("a += 1;"))
        self.assertIn("increment", operator_gloss("a++ + ++b"))

    def test_paired_analysis_counts_original_queries_once(self):
        qfeatures = {f"q{i}": {"n_gold": i, "question_words": 10, "code_chars": 25,
                               "domain_cue": True, "vague_reference": False} for i in (1, 2, 3)}
        rows = []
        for variant in VARIANTS:
            for method in METHODS:
                for qid in qfeatures:
                    value = 1.0 if variant == "explicit_one_sentence" else 0.5
                    rows.append({"variant": variant, "method": method, "query_id": qid,
                                 **{metric: value for metric in METRICS}})
        result = analyze(rows, qfeatures)
        effect = next(r for r in result["paired_contrasts"] if r["variant"] == "explicit_one_sentence"
                      and r["method"] == "dense" and r["metric"] == "recall@10")
        self.assertEqual(effect["n"], 3)
        self.assertEqual((effect["wins"], effect["losses"], effect["ties"]), (3, 0, 0))
        self.assertEqual(effect["mean_diff"], 0.5)
        self.assertEqual(tuple(effect["ci95"]), (0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
