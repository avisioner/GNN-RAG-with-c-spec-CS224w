# Query Representation and Dataset Diagnostics

This is a fixed, paired diagnostic experiment on all 49 existing queries, not a new benchmark or a held-out test. Every natural-language source question already consists of a single question sentence (one final question mark, no internal sentence boundary). Consequently, the existing benchmark cannot establish that one-sentence queries are better than multi-sentence queries. The experiment separates information removal, formatting, explicitness, and verbosity as far as the available data allow.

## Design

Nine representations are evaluated with fixed BM25, Dense, 2:1 Hybrid RRF, and Dense + GraphExp. BGE-small, corpus text, clause aggregation, candidate budget 50, and graph parameters (top-5 seeds, one hop, lambda=0.5, references+parent+sibling) are unchanged. The original benchmark, graph, gold labels, and milestone results are not modified.

| Variant | What changes | Primary comparison |
|---|---|---|
| question_only | Original question; no code | Reference |
| question_code | Original question followed by original code | Information added vs question_only |
| code_only | Original code; no question | Information removed vs question_code |
| labeled_code | Same question and code, explicit labels and code fence | Formatting vs question_code |
| code_first | Same labeled fields in reverse order | Order vs labeled_code |
| explicit_one_sentence | Authored, operator-specific question plus unchanged code | Explicitness vs question_code |
| verbose_explicit | Same explicit question/code plus generic preamble and instruction | Verbosity/framing vs explicit_one_sentence |
| operator_gloss | Original question/code plus deterministic operator-spelling glossary | Symbol verbalization vs question_code |
| question_domain | C++23 specification prefix plus original question; no code | Domain framing vs question_only |

The 49 authored explicit questions are candidate paraphrases derived from the visible question and code. They contain no clause IDs, gold lists, expected-property metadata, or retrieved evidence. They have not received independent semantic review: extra operator-specific wording can narrow an underspecified task. Retained labels are therefore provisional for those two rewrite conditions. They must not silently replace canonical benchmark questions. Verbose framing changes both length and instruction context; it is not a pure causal test of sentence count.

## Overall Results

Recall/coverage are percentages. Coverage means every gold clause is present.

| Variant | Method | R@5 | Cov@5 | R@10 | Cov@10 | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| question_only | bm25 | 53.06 | 36.73 | 57.82 | 42.86 | 0.496 |
| question_only | dense | 60.88 | 42.86 | 74.15 | 57.14 | 0.581 |
| question_only | hybrid_2_1 | 65.82 | 48.98 | 69.90 | 51.02 | 0.601 |
| question_only | dense_graph_exp | 61.73 | 46.94 | 69.39 | 51.02 | 0.553 |
| question_code | bm25 | 44.90 | 34.69 | 52.89 | 38.78 | 0.444 |
| question_code | dense | 52.89 | 32.65 | 70.92 | 53.06 | 0.520 |
| question_code | hybrid_2_1 | 61.05 | 42.86 | 70.24 | 51.02 | 0.539 |
| question_code | dense_graph_exp | 53.23 | 34.69 | 71.26 | 55.10 | 0.517 |
| code_only | bm25 | 12.93 | 4.08 | 26.02 | 14.29 | 0.144 |
| code_only | dense | 16.33 | 4.08 | 22.62 | 6.12 | 0.156 |
| code_only | hybrid_2_1 | 9.86 | 2.04 | 20.92 | 8.16 | 0.116 |
| code_only | dense_graph_exp | 16.33 | 4.08 | 23.64 | 6.12 | 0.169 |
| labeled_code | bm25 | 50.51 | 34.69 | 60.37 | 36.73 | 0.497 |
| labeled_code | dense | 49.49 | 28.57 | 63.44 | 51.02 | 0.453 |
| labeled_code | hybrid_2_1 | 57.99 | 38.78 | 73.64 | 53.06 | 0.549 |
| labeled_code | dense_graph_exp | 53.57 | 34.69 | 66.50 | 53.06 | 0.473 |
| code_first | bm25 | 50.51 | 34.69 | 60.37 | 36.73 | 0.497 |
| code_first | dense | 45.41 | 26.53 | 63.10 | 46.94 | 0.436 |
| code_first | hybrid_2_1 | 53.91 | 32.65 | 74.15 | 55.10 | 0.531 |
| code_first | dense_graph_exp | 52.55 | 32.65 | 66.50 | 48.98 | 0.462 |
| explicit_one_sentence | bm25 | 54.93 | 32.65 | 79.42 | 59.18 | 0.637 |
| explicit_one_sentence | dense | 72.28 | 46.94 | 84.35 | 67.35 | 0.714 |
| explicit_one_sentence | hybrid_2_1 | 77.55 | 59.18 | 91.50 | 79.59 | 0.787 |
| explicit_one_sentence | dense_graph_exp | 70.58 | 48.98 | 84.35 | 67.35 | 0.703 |
| verbose_explicit | bm25 | 60.71 | 38.78 | 72.96 | 48.98 | 0.596 |
| verbose_explicit | dense | 68.54 | 46.94 | 80.10 | 59.18 | 0.661 |
| verbose_explicit | hybrid_2_1 | 70.24 | 46.94 | 88.44 | 75.51 | 0.728 |
| verbose_explicit | dense_graph_exp | 66.50 | 44.90 | 80.10 | 59.18 | 0.676 |
| operator_gloss | bm25 | 69.22 | 46.94 | 81.46 | 61.22 | 0.691 |
| operator_gloss | dense | 66.84 | 48.98 | 84.86 | 67.35 | 0.674 |
| operator_gloss | hybrid_2_1 | 80.10 | 59.18 | 91.16 | 79.59 | 0.772 |
| operator_gloss | dense_graph_exp | 66.67 | 48.98 | 84.69 | 69.39 | 0.663 |
| question_domain | bm25 | 55.10 | 36.73 | 71.77 | 53.06 | 0.565 |
| question_domain | dense | 55.27 | 32.65 | 75.68 | 53.06 | 0.514 |
| question_domain | hybrid_2_1 | 67.69 | 44.90 | 81.80 | 61.22 | 0.651 |
| question_domain | dense_graph_exp | 57.99 | 36.73 | 76.19 | 55.10 | 0.530 |

## Paired Changes in Dense Retrieval

Per-query differences, with 95% percentile bootstrap intervals (2,000 query resamples, seed 0). Intervals are exploratory and not adjusted for multiple comparisons; variants are not independent datasets.

| Variant vs control | Metric | Mean change (pp) | 95% interval (pp) | Wins/losses/ties |
|---|---|---:|---|---|
| question_code vs question_only | recall@10 | -3.23 | [-8.50, +2.04] | 3/6/40 |
| question_code vs question_only | complete_coverage@10 | -4.08 | [-10.20, +0.00] | 0/2/47 |
| code_only vs question_code | recall@10 | -48.30 | [-60.20, -36.05] | 1/32/16 |
| code_only vs question_code | complete_coverage@10 | -46.94 | [-61.22, -32.65] | 0/23/26 |
| labeled_code vs question_code | recall@10 | -7.48 | [-15.31, -0.34] | 3/9/37 |
| labeled_code vs question_code | complete_coverage@10 | -2.04 | [-10.20, +6.12] | 2/3/44 |
| code_first vs labeled_code | recall@10 | -0.34 | [-5.10, +4.42] | 3/3/43 |
| code_first vs labeled_code | complete_coverage@10 | -4.08 | [-10.20, +0.00] | 0/2/47 |
| explicit_one_sentence vs question_code | recall@10 | +13.44 | [+3.57, +24.15] | 15/3/31 |
| explicit_one_sentence vs question_code | complete_coverage@10 | +14.29 | [+2.04, +28.57] | 10/3/36 |
| verbose_explicit vs explicit_one_sentence | recall@10 | -4.25 | [-9.18, +0.34] | 1/6/42 |
| verbose_explicit vs explicit_one_sentence | complete_coverage@10 | -8.16 | [-18.37, +2.04] | 1/5/43 |
| operator_gloss vs question_code | recall@10 | +13.95 | [+6.80, +22.11] | 15/1/33 |
| operator_gloss vs question_code | complete_coverage@10 | +14.29 | [+6.12, +24.49] | 7/0/42 |
| question_domain vs question_only | recall@10 | +1.53 | [-5.62, +8.85] | 9/7/33 |
| question_domain vs question_only | complete_coverage@10 | -4.08 | [-16.33, +8.16] | 4/6/39 |

## Original-Query Features and Confounding

These are observational subgroup summaries, not effects of changing query text. The 'domain cue' regex detects operator/type/sequence vocabulary; 'vague reference' detects words such as this, here, input, statement, updating, or modifying. These transparent heuristics are imperfect and can overlap. Length cutoffs are 12 question words and 80 code characters. Gold-size strata are reported to separate surface wording from the number of required clauses. Small groups should not be generalized.

| Gold stratum | Feature | Present | n | Dense R@10 | Dense Cov@10 |
|---|---|---|---:|---:|---:|
| all | domain_cue | False | 23 | 58.70 | 34.78 |
| all | domain_cue | True | 26 | 87.82 | 76.92 |
| all | question_words_le12 | False | 17 | 77.45 | 64.71 |
| all | question_words_le12 | True | 32 | 72.40 | 53.12 |
| gold_1 | domain_cue | True | 15 | 100.00 | 100.00 |
| gold_1 | question_words_le12 | False | 5 | 100.00 | 100.00 |
| gold_1 | question_words_le12 | True | 10 | 100.00 | 100.00 |
| gold_2 | domain_cue | False | 17 | 58.82 | 47.06 |
| gold_2 | domain_cue | True | 8 | 81.25 | 62.50 |
| gold_2 | question_words_le12 | False | 11 | 68.18 | 54.55 |
| gold_2 | question_words_le12 | True | 14 | 64.29 | 50.00 |
| gold_3_plus | domain_cue | False | 6 | 58.33 | 0.00 |
| gold_3_plus | domain_cue | True | 3 | 44.44 | 0.00 |
| gold_3_plus | question_words_le12 | False | 1 | 66.67 | 0.00 |
| gold_3_plus | question_words_le12 | True | 8 | 52.08 | 0.00 |

## Does the Graph Benefit Differ by Query Representation?

Same graph and propagation parameters for every variant; no graph setting was chosen from these results.

| Variant | Graph minus Dense Cov@10 (pp) | Wins/losses/ties |
|---|---:|---|
| question_only | -6.12 | 0/3/46 |
| question_code | +2.04 | 2/1/46 |
| code_only | +0.00 | 0/0/49 |
| labeled_code | +2.04 | 1/0/48 |
| code_first | +2.04 | 1/0/48 |
| explicit_one_sentence | +0.00 | 0/0/49 |
| verbose_explicit | +0.00 | 0/0/49 |
| operator_gloss | +2.04 | 1/0/48 |
| question_domain | +2.04 | 1/0/48 |

## Illustrative Rewrite Cases

The three largest positive and three largest negative Dense Recall@10 changes for explicit rewrites are selected after evaluation to expose both directions, not as representative samples.

- **specqa-021**, 1 gold clauses, change -100.00 pp. Original: Is a negative shift count undefined? Rewrite: Does a negative shift count in 1 << -1 cause undefined behavior in C++23?
- **specqa-011**, 2 gold clauses, change -50.00 pp. Original: What is the behavior of updating a on both sides of this return statement? Rewrite: What behavior does C++23 specify for the two compound assignments to a in (a += b) + (a += n)?
- **specqa-036**, 2 gold clauses, change -50.00 pp. Original: For this overloaded shift, is the left operand sequenced before the right? Rewrite: For the overloaded operator expression make_c() << make_a(), is the left operand sequenced before the right?
- **specqa-034**, 2 gold clauses, change +100.00 pp. Original: What is the behavior of decreasing a and writing that result back to a? Rewrite: What behavior does C++23 specify for assignment combined with prefix decrement in a = --a?
- **specqa-045**, 2 gold clauses, change +100.00 pp. Original: What is the behavior of updating this member and writing the result back to it? Rewrite: What behavior does C++23 specify for assignment and postfix increment of the same member in sptr->a = sptr->a++?
- **specqa-048**, 2 gold clauses, change +100.00 pp. Original: What is the behavior of the two updates of i in this return statement? Rewrite: What behavior does C++23 specify for the two postfix increments in the shift expression i++ << i++?

## Dataset Improvement Decisions

1. Preserve the original task and code, and add reviewed alternative wording as a separate field. Require operator/type/reference clarity where essential, without naming gold clauses or revealing the outcome.

2. Build matched concise/verbose and plain-language/technical-term pairs. Keep code, assumptions, and intended evidence fixed. Have a reviewer check semantic equivalence and evidence sufficiency before using the same gold set.

3. Add code-dependent contrast sets: similar questions with different operators or types. Retain both easy and difficult cases. A question-only retriever can benefit from linguistic cues without demonstrating that it distinguishes the programs.

4. Expand beyond the 11 currently gold-labeled clauses and annotate the supporting paragraph or span. Complete clause coverage can count a non-supporting paragraph as correct, so it is not a full evidence-quality check.

5. Group all paraphrases and related code families into the same train/dev/test split. Select retrieval/query-processing settings on dev; report original, variant-average, and worst-variant results on frozen test. Do not count nine versions of 49 questions as 441 independent test items.

## Reproduction

Run `baseline/.venv/Scripts/python.exe experiments/query_diagnostics/run_experiments.py` from the project root. Weights are loaded offline from the existing cache at the recorded model revision. Document embeddings are cached under `baseline/.cache/query_diagnostics/`. Use `--analyze-only` to regenerate this report from saved per-query metrics, without models.

`experiment_plan.json` records the fixed conditions and source hashes; `query_inputs.jsonl` contains exact retrieval texts; `clause_scores.json` stores full scores; `ranked_results.jsonl` stores top-50 clauses; `per_query.csv` contains metrics/features; `analysis.json` contains all paired intervals and strata; `run_config.json` records package versions and cache checks. Existing milestone artifacts and benchmark records remain unchanged.
