# Operator Glossary Results

The production text baseline now supports an optional deterministic operator
glossary. A fresh model execution reproduces the diagnostic condition exactly:
all 49 input strings and all 49 Top-50 clause rankings match for each of BM25,
Dense and 2:1 Hybrid. The selected Hybrid achieves **91.16% Recall@10** and
**79.59% complete coverage@10 (39/49 queries)** on the existing benchmark.

This is an exploratory development result. The feature was selected after
comparing representations on these same 49 queries; it is not a new independent
test estimate. The historical [preliminary results](../RESULTS.md)
remain available as the original references.

## What changed

`--query-transform operator_gloss` appends a fixed list of descriptions to the
original question and complete code. For example:

```text
<original question>
<original code>
Operator spellings in the code: ++: increment; =: assignment spelling
```

The implementation reads only `query` and `code`. It does not use gold clause
IDs, difficulty, expected behavior, manual rewrites, corpus matches or model
outputs to construct the input. BM25, Dense and the optional reranker receive
the same transformed text. The BGE query instruction is applied separately.

The default remains `--query-transform none`. The glossary requires
`--query-mode query_code`, so question-only experiments cannot accidentally
include code-derived information. Canonical questions, programs and gold
labels are unchanged. Manually authored diagnostic rewrites are not adopted.

## Results on the unchanged 49 queries

RRF uses Dense:BM25 weights 2:1, constant 60 and 50 distinct-clause candidates.
BM25 uses the existing content preprocessing; Dense uses normalized BGE-small
embeddings with exact cosine ranking. Other settings are fixed between rows.

| Input | Retriever | Recall@5 | Recall@10 | Complete coverage@10 |
| --- | --- | ---: | ---: | ---: |
| Original question + code | BM25 | 44.90% | 52.89% | 38.78% |
| Original question + code + glossary | BM25 | 69.22% | 81.46% | 61.22% |
| Original question + code | Dense | 52.89% | 70.92% | 53.06% |
| Original question + code + glossary | Dense | 66.84% | 84.86% | 67.35% |
| Original question + code | 2:1 Hybrid | 61.05% | 70.24% | 51.02% |
| Original question + code + glossary | 2:1 Hybrid | 80.10% | 91.16% | 79.59% |

The paired Hybrid Recall@10 difference is **+20.92 percentage points**:
20 queries improve, 1 worsens and 28 tie. The exploratory paired bootstrap 95%
interval is **[+12.76, +30.27]** points, without multiple-comparison correction.
Complete coverage improves from 25/49 to 39/49: 15 wins, 1 loss and 33 ties.
These intervals come from the original paired diagnostic, whose rankings are
identical to this execution. See the [full diagnostic report](../../query_diagnostics/RESULTS.md)
for all nine conditions, per-query losses, strata and graph comparisons.

## Reproduce and verify

From the project root with the retained CPU environment and cached model:

```powershell
& baseline/specgraph_baselines/run_preliminary.ps1 -Offline -SkipRerank -QueryMode query_code -QueryTransform operator_gloss -DenseWeight 2 -EmbeddingRevision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a -OutputName operator_gloss_reproduction

& baseline/.venv/Scripts/python.exe experiments/query_diagnostics/verify_baseline_integration.py --run-dir results/baseline/operator_gloss_reproduction
```

Omit `-Offline` if weights have not yet been cached. Use a fresh output directory
to preserve recorded executions. No cross-encoder is used for these results.

The recorded execution is [this directory](.).
It contains metrics, clause rankings, representative evidence chunk IDs, exact
query input strings, corpus mappings, audits, input/package configuration,
source snapshots and [reproduction_verification.json](reproduction_verification.json).
The configuration records the glossary version, explicit embedding revision,
input hashes and exact query-input file hash.

Verification checks every transformed input, all three methods' complete Top-50
rankings, unchanged gold labels, distinct-clause budgets, valid chunk-to-clause
evidence mappings, corpus/benchmark identity and source snapshot hashes.
The baseline regression suite has 21 passing tests; the diagnostic suite has 5.
Repository validation also passes the 25 graph tests; see the
[combined results](../../README.md) for validation commands. Repository
organization updates graph result paths without changing graph parameters.

## Limits and dataset follow-up

The versioned `operator_gloss_v1` mapping preserves the diagnostic experiment.
It recognizes a small set of operator spellings with regular expressions; it
does not parse C++, remove comments/string literals or resolve overloaded
operators. Some punctuation can have another syntactic role. Its vocabulary is
incomplete, and broader programs need separate evaluation. A parser-aware
revision should be a new condition, retaining v1 for comparison.

No transformed query exceeds the model's 512-token limit in this execution.
Nine corpus chunks exceed that limit and are truncated, as in the prior runs.
Clause-level recall does not prove that the selected representative chunk
contains the needed paragraph. Generation and answer correctness are not tested.

The next dataset work should independently review task scope, assumptions and
supporting evidence, then test a reviewed verbosity-by-terminology design on
new development families. Keep related variants in one split and freeze a new
test set before evaluating the selected pipeline. The
[query review guide](../../../docs/dataset/QUERY_REVIEW_GUIDE.md) and
[annotation template](../../../docs/dataset/query_review_template.json) describe
the proposed workflow. Existing rewrite candidates retain pending review status.
