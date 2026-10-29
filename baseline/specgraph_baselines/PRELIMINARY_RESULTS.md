# Preliminary Text Retrieval Results for SpecGraph-RAG

The updated baselines improve retrieval on the current 49-query N4950 benchmark. **Question-only Dense retrieval achieves 74.15% Recall@10 and 57.14% complete clause coverage@10**, compared with 70.92% and 53.06% when the code is appended to the question. For a smaller retrieval budget, **question-only weighted RRF with a Dense:BM25 ratio of 2:1 achieves 65.82% Recall@5 and 48.98% complete coverage@5**.

These are exploratory results on the available benchmark. They provide preliminary reference points; the configurations were compared on these same 49 queries and have not been validated on a separate held-out test set.

## Experimental Setup

The corpus contains 454 chunks representing 82 distinct C++23 N4950 clauses. The benchmark contains 49 labeled queries: 15 single-clause, 17 two-clause, and 17 multi-hop questions. All gold clause IDs are present in the corpus.

Documents contain only their title and text. Retrieval inputs are either the original question plus code (`query_code`) or the question alone (`question`). Concepts, gold labels, difficulty, expected properties, and source URLs are excluded from retrieval inputs. The corpus and benchmark labels were unchanged throughout the experiments.

| Component | Configuration |
| --- | --- |
| Dense encoder | `BAAI/bge-small-en-v1.5`, with its query instruction |
| Similarity | Exact cosine ranking over normalized chunk embeddings |
| BM25 | k1=1.5, b=0.75; IDF = log(1 + (N-df+0.5)/(df+0.5)) |
| Lexical preprocessing | General English stopword removal; single-letter variable and delimiter filtering; C++ operators retained |
| Candidate budget | 50 distinct clauses per retriever; 50 after fusion |
| RRF constant | 60 |
| Dense:BM25 weights compared | 1:1, 2:1, 3:1, for both input modes |
| Original cross-encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Runtime | Python 3.12.14, CPU PyTorch 2.15.0+cpu, Sentence Transformers 5.7.0, 4 CPU threads |

Each text retriever first ranks all chunks, then collapses its ranking to distinct clauses using the best-ranked chunk. Candidate budgets and evaluation k values count clauses. Duplicate chunks cannot consume multiple positions or vote repeatedly in clause-level fusion. This aggregation can still favor clauses with more chunks by giving them more opportunities to score well.

Weighted fusion assigns a clause the score:

`score(c) = 1 / (60 + rank_BM25(c)) + w / (60 + rank_Dense(c))`

Here, `w` is 1, 2, or 3. A clause absent from a branch's saved top-50 candidate list receives zero contribution from that branch. The fused ranking is truncated to 50 clauses, with stable clause IDs breaking ties. We replayed this fixed grid using saved rankings, without additional inference or training. Equal-weight replay exactly reproduces the original hybrid clause rankings.

Metrics are macro-averaged over queries. Recall measures the fraction of each query's gold clauses retrieved. Complete coverage is one only when every gold clause is retrieved. Binary nDCG measures ranking quality. Recall and coverage below are percentages; nDCG ranges from 0 to 1.

## Original Question Plus Code Results

| Method | Recall@5 | Recall@10 | Coverage@5 | Coverage@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25, preprocessed | 44.90% | 52.89% | 34.69% | 38.78% | 0.44448 |
| Dense | 52.89% | 70.92% | 32.65% | 53.06% | 0.52045 |
| Hybrid RRF, 1:1 | 54.93% | 69.39% | 38.78% | 51.02% | 0.54178 |
| Hybrid RRF, 2:1 | 61.05% | 70.24% | 42.86% | 51.02% | 0.53890 |
| Hybrid RRF, 3:1 | 61.73% | 70.58% | 42.86% | 53.06% | 0.53428 |
| Original Hybrid + Cross-Encoder | 44.05% | 55.78% | 30.61% | 40.82% | 0.41807 |

Ratios are always **Dense:BM25**. With question plus code, increasing the dense weight to 3 improves Recall@5 by 6.80 percentage points relative to equal-weight RRF, and coverage@5 by 4.08 points. Pure Dense remains slightly higher in Recall@10, while equal-weight RRF has the highest nDCG@10 in this input mode. No configuration dominates every metric.

The earlier unfiltered BM25 run achieved 26.19% Recall@10 and 16.33% coverage@10. It remains available as a preprocessing comparison in `results_bm25/`; it was not overwritten.

## Question-Only Ablation

| Method | Recall@5 | Recall@10 | Coverage@5 | Coverage@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25, preprocessed | 53.06% | 57.82% | 36.73% | 42.86% | 0.49593 |
| Dense | 60.88% | **74.15%** | 42.86% | **57.14%** | 0.58112 |
| Hybrid RRF, 1:1 | 61.91% | 70.24% | 46.94% | 53.06% | 0.59858 |
| Hybrid RRF, 2:1 | **65.82%** | 69.90% | **48.98%** | 51.02% | 0.60147 |
| Hybrid RRF, 3:1 | 64.80% | 71.60% | 46.94% | 53.06% | **0.60700** |

Removing the raw code improves Dense Recall@10 by 3.23 percentage points and complete coverage from 26/49 to 28/49 queries. This suggests that directly appending code is not the best input representation for this text encoder on this benchmark. It does not establish that code is generally unhelpful: code can be necessary to distinguish semantic cases, and structured code representations were not evaluated.

Question-only 2:1 RRF provides the highest observed Recall@5 and coverage@5. Compared with the original question-plus-code equal-weight hybrid, these metrics increase by 10.88 and 10.20 percentage points, respectively. That combined improvement includes both the input change and the weighting change. Within the question-only setting, changing 1:1 to 2:1 improves Recall@5 by 3.91 points and coverage@5 by 2.04 points.

For preliminary reporting, use question-only Dense as the @10 reference and question-only 2:1 RRF as the @5 reference. Retain the original input-mode results for comparison. The standard command-line defaults remain question plus code and equal weighting, so the experimental configurations must be selected explicitly.

## Performance by Difficulty

The table below shows the two question-only reference configurations. Each metric is a percentage.

| Difficulty | Queries | Dense Recall@5 | Dense Recall@10 | Dense Coverage@10 | 2:1 RRF Recall@5 | 2:1 RRF Recall@10 | 2:1 RRF Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single clause | 15 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Two clause | 17 | 52.94 | 76.47 | 64.71 | 61.76 | 67.65 | 47.06 |
| Multi-hop | 17 | 34.31 | 49.02 | 11.76 | 39.71 | 45.59 | 11.76 |

Both configurations retrieve the gold for every single-clause query within five results. Multi-hop complete coverage remains 2/17 at k=10, despite partial recall gains. The overall improvements therefore do not resolve the main evidence-completeness problem on difficult queries. Full difficulty and gold-set-size breakdowns are included in the saved metrics.

## Cross-Encoder Result and Candidate Coverage

The original MiniLM cross-encoder reduced Recall@10 from 69.39% to 55.78%. Its equal-weight question-plus-code hybrid candidate set had 98.30% Recall@50 and complete coverage for 47/49 queries. The loss occurs in final ordering rather than the absence of gold from most candidate sets.

This is not explained solely by selecting the wrong paragraph. For `specqa-001`, the gold clause `expr.pre.incr` has a single chunk and moves from rank 1 to rank 11 after reranking. For `specqa-002`, the gold moves from hybrid rank 2 to reranked rank 14. Domain mismatch between generic passage reranking and C++ specification relevance is one possible explanation, but has not been independently established.

The cross-encoder result is retained as a negative result. It was not rerun for the new input modes or weights and should not be interpreted as a reranker evaluation for those configurations. Weighted replay changes clause rankings only; its exported evidence chunks are selected from the source retriever with the highest weighted clause-rank contribution. No cross-encoder consumes those replayed evidence selections.

For question-only retrieval, Dense has 97.62% candidate Recall@50, while equal-weight Hybrid has 98.64%. Question-only BM25 has only 84.01%, despite its better @10 result. Candidate coverage and final ordering should therefore be reported separately.

## Interpretation and Limitations

The input modes and three weight ratios were compared on the same 49 queries after examining the initial baseline results. All planned outcomes are reported; no model was trained on the benchmark. These comparisons are exploratory configuration selection, not held-out generalization estimates. A difference of two completely covered queries is useful preliminary evidence, but not proof of a stable advantage.

Evaluation is at clause granularity: any returned chunk belonging to a gold clause counts as a hit, even if that paragraph does not support the answer. Nine of 454 document inputs exceed BGE's 512-token limit and are truncated; query inputs do not exceed the limit. These limitations apply when interpreting evidence completeness and downstream generation quality.

## Reproduction and Saved Artifacts

The CPU environment is in `baseline/.venv/`; model weights are cached under `baseline/.cache/huggingface/`. Package versions are pinned in `requirements-preliminary-lock.txt`, and the observed model commits are recorded in `preliminary_summary/model_revisions.json`. The query-only execution and original execution also include source snapshots and input hashes.

To rerun the question-only baselines, from the project root:

```powershell
& baseline/specgraph_baselines/run_preliminary.ps1 -Offline -SkipRerank -QueryMode question -DenseWeight 1 -OutputName results_question_only_reproduction
```

To directly run the question-only 2:1 configuration:

```powershell
& baseline/specgraph_baselines/run_preliminary.ps1 -Offline -SkipRerank -QueryMode question -DenseWeight 2 -OutputName results_question_weighted_reproduction
```

Replay the fixed weight grid from the existing rankings without loading models:

```powershell
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/replay_fusion.py --source baseline/specgraph_baselines/results_preliminary_bge_small --output-dir baseline/specgraph_baselines/results_ablation_query_code
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/replay_fusion.py --source baseline/specgraph_baselines/results_question_only --output-dir baseline/specgraph_baselines/results_ablation_question
```

Rebuild the comparison tables:

```powershell
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/summarize_preliminary.py baseline/specgraph_baselines/results_bm25 baseline/specgraph_baselines/results_ablation_query_code baseline/specgraph_baselines/results_ablation_question
```

| Artifact | Contents |
| --- | --- |
| `results_preliminary_bge_small/` | Original question-plus-code model execution, including the cross-encoder |
| `results_question_only/` | Actual question-only BM25, Dense, and equal-weight Hybrid execution |
| `results_ablation_query_code/` | Three fusion weights replayed from the original input mode |
| `results_ablation_question/` | Three fusion weights replayed from question-only inputs |
| `preliminary_summary/comparison.csv` | All 12 configurations, including the original unfiltered BM25 comparison |
| `preliminary_summary/comparison.json` | Full metrics, difficulty breakdowns, configurations, and provenance |

Each result directory contains clause rankings, aligned evidence chunk IDs, metrics, and configuration metadata. Eighteen regression checks pass, including weighted-fusion behavior, input isolation, duplicate handling, and metric calculations. All saved ablation rankings contain 50 distinct clauses per query and valid evidence mappings. The Chroma backend was not used in these experiments.

## Suggested Preliminary Results Paragraph

> We evaluated text retrieval baselines on 49 annotated N4950 queries over 454 chunks representing 82 specification clauses. A small ablation comparing question-only and question-plus-code inputs, with Dense:BM25 fusion weights of 1:1, 2:1, and 3:1, improved our preliminary reference results. Question-only BGE-small retrieval achieved 74.15% Recall@10 and 57.14% complete clause coverage@10, compared with 70.92% and 53.06% for question-plus-code inputs. At k=5, question-only 2:1 weighted reciprocal rank fusion achieved 65.82% recall and 48.98% complete coverage. A generic MS MARCO cross-encoder reduced performance in the original configuration. Multi-hop complete coverage remained low, indicating that stronger overall retrieval did not consistently recover full evidence sets for difficult queries. These results are exploratory comparisons on the available benchmark rather than held-out test estimates.
