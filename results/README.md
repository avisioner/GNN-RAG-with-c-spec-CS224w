# Results

All reported experiments use the unchanged 49-query benchmark and 454 chunks
representing 82 N4950 clauses. Metrics count distinct clauses and average over
queries. Complete coverage means every gold clause is retrieved. These are
exploratory development comparisons: the same queries informed configuration
choices, and no independent test set has validated the selected pipeline.

## Main comparison

Hybrid ratios are always **Dense:BM25**. The glossary is deterministic query
preprocessing that retains the original question and full code.

| Input | Method | Recall@5 | Recall@10 | Complete coverage@10 |
| --- | --- | ---: | ---: | ---: |
| Original question + code | BM25 | 44.90% | 52.89% | 38.78% |
| Original question + code | Dense | 52.89% | 70.92% | 53.06% |
| Original question + code | Hybrid 2:1 | 61.05% | 70.24% | 51.02% |
| Original question only | Dense | 60.88% | 74.15% | 57.14% |
| Original question only | Hybrid 2:1 | 65.82% | 69.90% | 51.02% |
| Original question only | Dense + GraphExp | 61.73% | 69.39% | 51.02% |
| Original question + code + glossary | BM25 | 69.22% | 81.46% | 61.22% |
| Original question + code + glossary | Dense | 66.84% | 84.86% | 67.35% |
| Original question + code + glossary | **Hybrid 2:1** | **80.10%** | **91.16%** | **79.59%** |

## Findings

Automatic operator descriptions improve Hybrid Recall@10 by **20.92 percentage
points** over the same question+code pipeline: 20 query wins, 1 loss and 28 ties.
Complete coverage increases from 25/49 to 39/49. The paired bootstrap 95%
interval for the recall gain is [+12.76, +30.27] points, without correction for
multiple comparisons. A fresh production run matches every diagnostic Top-50
ranking for all three text methods on all 49 queries.

Representation matters, but the experiment does not establish that shorter
questions are better. All original questions are already single-sentence
questions. Manual explicit rewrites also help, but remain unreviewed candidates;
their intervention changes terminology, context and sometimes task scope.

The fixed question-only graph expansion does not improve complete coverage@10
over Dense: 0 wins, 3 losses, 46 ties. Its relative effect changes with query
representation; graph comparisons for all nine inputs are retained below.
The original generic cross-encoder is also retained as a negative result.

Keep the existing benchmark as a reference. Review ambiguous tasks and their
evidence, broaden topic/code-family coverage, and freeze a new family-separated
test set before judging generalization. Only 11 distinct clauses appear in the
current gold sets. Clause hits do not guarantee paragraph support or correct
answers; nine source chunks exceed the encoder's token limit.

## Detailed results and saved runs

| Experiment | Results | Artifacts |
| --- | --- | --- |
| Original text baselines and fusion grid | [Text retrieval](baseline/RESULTS.md) | [12-configuration comparison](baseline/summary/comparison.csv), [BM25](baseline/bm25/), [question+code](baseline/query_code/), [question-only](baseline/question_only/), [code fusion](baseline/fusion_query_code/), [question fusion](baseline/fusion_question_only/) |
| Automatic operator glossary in the production baseline | [Operator glossary](baseline/operator_gloss/RESULTS.md) | [Metrics](baseline/operator_gloss/metrics.json), [rankings](baseline/operator_gloss/ranked_results.jsonl), [verification](baseline/operator_gloss/reproduction_verification.json) |
| Nine paired query representations, four retrieval methods | [Overview](query_diagnostics/README.md), [full results](query_diagnostics/RESULTS.md) | [Per-query metrics](query_diagnostics/per_query.csv), [exact inputs](query_diagnostics/query_inputs.jsonl), [full scores](query_diagnostics/clause_scores.json), [pending rewrite review](query_diagnostics/rewrite_review.csv) |
| Graph expansion milestone and ablations | [Graph results](milestone/RESULTS.md) | [Per-query metrics](milestone/per_query.csv), [data audit](milestone/data_validation.md), [configuration](milestone/run_config.json) |

## Reproduction and provenance

- [Baseline setup and commands](../baseline/specgraph_baselines/README.md)
- [Query experiment and no-model verification](../experiments/query_diagnostics/README.md)
- [Graph reproduction](../graph/specgraph_graph/README.md)
- [Dataset review guide](../docs/dataset/QUERY_REVIEW_GUIDE.md)

Historical rankings, scores, configurations and source snapshots are preserved.
Their metadata records the original execution paths and hashes; those paths may
predate the current layout. [layout_manifest.json](layout_manifest.json) maps
moved paths and records hashes of the unchanged run artifacts. Use the current
scripts and READMEs for reproduction; frozen source snapshots document what ran.

```powershell
& baseline/.venv/Scripts/python.exe scripts/check_repository.py
& baseline/.venv/Scripts/python.exe experiments/query_diagnostics/verify_baseline_integration.py --check-only
```

Repository validation passes **51 tests**: 21 baseline, 5 query-diagnostic and
25 graph checks. The standard-library audit additionally verifies local links,
relocated artifact hashes, saved metrics, evidence mappings and diagnostic
input/ranking counts. Re-run the suites with the project environment (`pytest`
is needed only for the graph suite):

```powershell
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/retrieval_baselines.py --self-test
& baseline/.venv/Scripts/python.exe -m unittest discover -s experiments/query_diagnostics -p test_query_diagnostics.py -v
& baseline/.venv/Scripts/python.exe -m pytest graph/specgraph_graph/tests -q
```
