# Query Representation Diagnostics

This experiment asks whether query representation, rather than only retrieval
architecture, explains some of the current benchmark results. It leaves the
canonical dataset and the graph milestone outputs unchanged.

Start with [findings and dataset decisions](../../results/query_diagnostics/README.md),
then the [full paired report](../../results/query_diagnostics/RESULTS.md).

The glossary condition is now available through the normal baseline CLI. A
fresh execution matches all 49 Top-50 rankings for each text method; see
[integration results](../../results/baseline/operator_gloss/RESULTS.md).
The [dataset review guide](../../docs/dataset/QUERY_REVIEW_GUIDE.md) proposes
independent review and the next controlled wording experiment.

## Fixed Experiment

Each of the 49 queries has nine retrieval inputs: original question, original
question+code, code only, labeled question+code, labeled code first, an explicit
one-sentence question+code, a verbose version of that explicit question+code,
question+code with a deterministic operator glossary, and question with a C++23
domain prefix. The glossary describes token spellings, not program behavior.

All inputs are evaluated with BM25, BGE-small Dense, 2:1 weighted RRF, and the
existing Dense + GraphExp settings. There is no parameter search. The runner
first verifies that recomputed original question-only Dense and BM25 rankings
match all 49 saved references. The exact teammate score cache is retained for
the question-only graph reference.

The nine inputs are not nine independent tasks. Paired changes and bootstrap
intervals are calculated over the 49 original query IDs. Existing-question
features are stratified by gold-set size to expose confounding.

## Candidate Rewrite Review

`explicit_questions.json` contains assistant-authored diagnostic questions
derived from the visible question and program. They are not independently
reviewed replacements for the benchmark. The builder only accepts the question,
code, and candidate wording; it has no access to labels, corpus text or retrieval
results. Every condition that uses code retains the complete original program.

Candidate rewrites verbalize operators and can repeat snippets, add domain
framing or narrow an otherwise underspecified question. Their effect cannot be
attributed to sentence length alone. Review semantic scope, code correspondence,
assumptions, and gold validity before adopting any candidate.

The current original questions are already single-sentence questions. The
explicit vs verbose comparison changes framing and length; it is not a clean
intervention on sentence count. A future length-only paraphrase study should
use independently reviewed matched pairs.

## Run

From the project root, with the existing local environment and model cache:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& baseline/.venv/Scripts/python.exe -m unittest discover -s experiments/query_diagnostics -p test_query_diagnostics.py -v
& baseline/.venv/Scripts/python.exe experiments/query_diagnostics/run_experiments.py
```

The runner loads the recorded BGE model revision offline, computes document
embeddings once, and caches them under `baseline/.cache/query_diagnostics/`.
All exact input strings and the fixed experiment plan are saved before inference.
No paid API or additional model is used.

Regenerate statistical analysis and the English report without models:

```powershell
& baseline/.venv/Scripts/python.exe experiments/query_diagnostics/run_experiments.py --analyze-only
```

Verify the saved production glossary run without models:

```powershell
& baseline/.venv/Scripts/python.exe experiments/query_diagnostics/verify_baseline_integration.py --check-only
```

## Outputs

Results are written only under `results/query_diagnostics/`:

| File | Purpose |
| --- | --- |
| `RESULTS.md` | English results, paired comparisons, confounding and dataset recommendations |
| `experiment_plan.json` | Fixed conditions, parameters, controls, review status and source hashes |
| `query_inputs.jsonl` | Exact retrieval input for every query/variant; no labels |
| `rewrite_review.csv` | Optional semantic-review checklist for candidate rewrites |
| `ranked_results.jsonl` | Top-50 clause rankings for every query/variant/method |
| `clause_scores.json` | Full Dense and BM25 scores over the 82 clauses |
| `per_query.csv` | Per-query metrics and transparent original-query features |
| `analysis.json` | Aggregate metrics, paired intervals, feature strata and graph-vs-Dense effects |
| `run_config.json` | Model revision, package versions, input/cache hashes and reproduction checks |

The interpretation should guide better coverage and clearer task definitions,
not removal of difficult items merely because a model scores poorly on them.
Keep paraphrases and related code families in the same future data split.
