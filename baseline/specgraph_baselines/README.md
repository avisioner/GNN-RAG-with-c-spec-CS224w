# Text Retrieval Baselines

BM25, BGE-small Dense retrieval, weighted clause-level RRF and optional MiniLM
cross-encoder reranking. Read [Results](../../results/README.md) for the combined
findings, [text results](../../results/baseline/RESULTS.md) for the original
experiments, and [glossary results](../../results/baseline/operator_gloss/RESULTS.md)
for the latest input improvement. All saved runs live in `results/baseline/`.

## Setup

The retained local environment is `baseline/.venv/`; downloaded model weights
are in `baseline/.cache/huggingface/`. Do not recreate an existing environment.
For a new environment:

```powershell
python -m venv baseline/.venv
& baseline/.venv/Scripts/python.exe -m pip install -r baseline/specgraph_baselines/requirements-preliminary-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` lists compatible direct dependencies; the lock file records
the actual CPU experiment environment. Observed model commits are in
[model_revisions.json](../../results/baseline/summary/model_revisions.json).
BM25 alone uses only the standard library.

## Run

Run from the project root. The glossary configuration preserves the question
and complete code, appending a fixed operator-spelling glossary:

```powershell
& baseline/specgraph_baselines/run_preliminary.ps1 -Offline -SkipRerank -QueryMode query_code -QueryTransform operator_gloss -DenseWeight 2 -EmbeddingRevision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a -OutputName operator_gloss_reproduction
```

This writes BM25, Dense and 2:1 RRF results at k=5, 10 and 50 to
`results/baseline/operator_gloss_reproduction/`. Omit `-Offline` when weights
need downloading. Use a new output name to preserve saved experiments.
The wrapper defaults to `results/baseline/reproduction/`.

The original question-only reference remains reproducible:

```powershell
& baseline/specgraph_baselines/run_preliminary.ps1 -Offline -SkipRerank -QueryMode question -DenseWeight 2 -OutputName question_only_reproduction
```

For lexical retrieval only:

```powershell
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/retrieval_baselines.py --methods bm25 --query-mode question --lexical-mode content --output-dir results/baseline/bm25_reproduction
```

`--query-transform none` is the default. The optional `operator_gloss` requires
`query_code` and describes spellings, not C++ semantics. It does not parse code
or resolve overloads, and can match comments/literals. Query inputs exclude
annotations and are exported exactly in `query_inputs.jsonl` with a file hash.
`--embedding-revision` optionally pins a model commit. `-SkipRerank` omits the
cross-encoder; its negative original result is retained, but it was not rerun
for the glossary. All code-based retrievers receive the same transformed input.

## Data and ranking contract

Default corpus and benchmark paths resolve from the repository root:
`data/processed/chunks.jsonl` and `data/benchmark/spec_benchmark_v1.jsonl`.
The reader handles ordinary JSONL, JSON arrays and the corpus's multiline
object stream. Several chunks can share a clause. Legacy clause-only input is
supported through `--clauses`, an alias for `--chunks`.

Documents contain only title and text. Queries contain question+code by default
or only the question with `--query-mode question`. Gold, difficulty, expected
properties and cross-reference metadata do not enter retrieval inputs.
`--graph data/graph/spec_graph.json` validates mappings and exports node IDs;
the graph is not used for text ranking.

BM25 uses positive Robertson IDF, k1=1.5, b=0.75 and C++ tokenization.
`--lexical-mode content` filters stopwords, single-letter variables and structural
delimiters; `raw` preserves tokens. Dense uses normalized embeddings and exact
cosine ranking, with a query-only instruction and four CPU threads.

Rank all chunks, collapse to distinct clauses using the best-ranked chunk,
then apply the candidate budget (50 by default). RRF combines clause ranks with
constant 60 and `--dense-weight` relative to BM25's weight of 1. Chunk RRF
separately selects evidence. The optional cross-encoder uses the same Hybrid
clause candidates and one representative chunk per clause.

Metrics are macro Recall, Precision, complete coverage and binary nDCG, grouped
by difficulty and gold-set size. Precision divides by requested k. Duplicate
clause rankings are rejected; empty gold sets are excluded. A clause hit does
not certify paragraph support. The optional Chroma backend (`chromadb>=1,<2`)
is not used for reported results; it retrieves all chunks before collapse.

## Checks and no-model replay

```powershell
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/retrieval_baselines.py --dry-run --graph data/graph/spec_graph.json
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/retrieval_baselines.py --self-test
& baseline/.venv/Scripts/python.exe experiments/query_diagnostics/verify_baseline_integration.py --check-only

& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/replay_fusion.py --source results/baseline/query_code --output-dir results/baseline/fusion_query_code_reproduction
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/replay_fusion.py --source results/baseline/question_only --output-dir results/baseline/fusion_question_only_reproduction
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/summarize_preliminary.py results/baseline/bm25 results/baseline/fusion_query_code results/baseline/fusion_question_only --output-dir results/baseline/summary_reproduction
```

Replay evaluates Dense:BM25 weights 1:1, 2:1 and 3:1 from saved candidates.
Its evidence comes from the branch with the larger weighted clause-rank
contribution, Dense breaking ties. This differs from online chunk-RRF evidence
selection and does not change clause metrics. No reranker uses replayed evidence.
