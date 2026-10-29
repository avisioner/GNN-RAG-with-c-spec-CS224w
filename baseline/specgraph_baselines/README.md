# SpecGraph-RAG Text Retrieval Baselines

The completed preliminary experiments compare BM25, BGE-small Dense retrieval,
weighted BM25–Dense RRF, and an optional MiniLM cross-encoder. Read
[PRELIMINARY_RESULTS.md](PRELIMINARY_RESULTS.md) for the English report and all
results. [preliminary_summary/comparison.csv](preliminary_summary/comparison.csv)
contains all 12 evaluated configurations.

| Preliminary reference | Recall | Complete coverage |
| --- | ---: | ---: |
| Question-only Dense, k=10 | 74.15% | 57.14% |
| Question-only 2:1 RRF, k=5 | 65.82% | 48.98% |

These configurations were compared on the same 49 queries. They are exploratory
results rather than held-out test estimates. The original cross-encoder result
is preserved, including its performance decrease.

## Run the Recommended Configuration

Run from the project root. The existing CPU environment and cached model weights
are retained for offline use:

```powershell
& baseline/specgraph_baselines/run_preliminary.ps1 -Offline -SkipRerank -QueryMode question -DenseWeight 2 -OutputName results_reproduction
```

This outputs BM25, Dense, and 2:1 Hybrid results at k=5, 10, and 50. Use a new
output name to preserve the recorded experiments. Omit `-Offline` if models need
to be downloaded. `-DenseWeight 1` reproduces equal-weight fusion. Select
`-QueryMode query_code` for the original question-plus-code input. Omitting
`-SkipRerank` also runs the cross-encoder, which is not recommended as the current
reference configuration.

Validation and tests, without downloading or running models:

```powershell
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/retrieval_baselines.py --dry-run --graph data/graph/spec_graph.json
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/retrieval_baselines.py --self-test
```

BM25 alone requires only the Python standard library:

```powershell
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/retrieval_baselines.py --methods bm25 --query-mode question --lexical-mode content --output-dir baseline/specgraph_baselines/results_bm25_reproduction
```

For a new environment, `requirements.txt` specifies compatible direct
dependencies; `requirements-preliminary-lock.txt` pins the actual CPU experiment
environment. The pinned `torch` wheel requires the PyTorch CPU index:

```powershell
python -m venv baseline/.venv
& baseline/.venv/Scripts/python.exe -m pip install -r baseline/specgraph_baselines/requirements-preliminary-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Do not recreate `.venv` when using the existing local environment. Model commits
are recorded in `preliminary_summary/model_revisions.json`; the retained offline
cache contains those snapshots.

## Data and Ranking Contract

Default input paths are resolved relative to the script; explicit paths and
output paths are resolved relative to the working directory.

| Input | Default file |
| --- | --- |
| Corpus | `data/processed/chunks.jsonl` |
| Queries | `data/benchmark/spec_benchmark_v1.jsonl` |
| Optional graph validation | `--graph data/graph/spec_graph.json` |

The corpus currently contains multiline JSON objects. The reader also accepts
ordinary JSONL and JSON arrays, including a UTF-8 BOM. Each record has a unique
`chunk_id`, a `clause_id`, and text; several chunks can share a clause. Legacy
clause-only records are supported through `--clauses` (an alias for `--chunks`).

Documents use only `title + text`. Queries use `query + code` by default, or
only `query` with `--query-mode question`. Annotation fields and cross-reference
metadata do not enter model inputs. References already present in the source
text remain part of the text. The graph is used for validation and ID export,
not for text retrieval ranking.

BM25 uses positive Robertson IDF, k1=1.5, b=0.75, and C++ operator-aware
tokenization. `--lexical-mode content` removes general stopwords, single-letter
variables and structural delimiters; `raw` preserves the original tokenization.
Dense uses normalized BGE embeddings and exact cosine ranking. Its query
instruction is applied to queries only. `--threads 4` controls CPU threads.

Both retrievers rank all chunks and collapse to distinct clauses before applying
the 50-clause candidate budget. RRF combines clause ranks with constant 60 and
`--dense-weight` relative to BM25's weight of 1. The online pipeline separately
selects evidence using chunk RRF. Optional cross-encoder reranking uses exactly
the Hybrid candidates and one representative chunk per clause.

Metrics are macro Recall, Precision, complete coverage and binary nDCG, grouped
by difficulty and gold-set size. Precision divides by requested k. Duplicate
ranked clause IDs are rejected; empty gold lists are excluded from evaluation.
Clause recall does not certify paragraph evidence correctness.

An optional persistent backend remains available with `--dense-backend chroma`
and `chromadb>=1.0,<2`; it was not used for the reported experiments. It stores
chunk IDs and retrieves all chunks before clause collapse.

## Replay and Summarize Existing Experiments

No models are loaded by these commands:

```powershell
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/replay_fusion.py --source baseline/specgraph_baselines/results_preliminary_bge_small --output-dir baseline/specgraph_baselines/results_ablation_query_code
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/replay_fusion.py --source baseline/specgraph_baselines/results_question_only --output-dir baseline/specgraph_baselines/results_ablation_question
& baseline/.venv/Scripts/python.exe baseline/specgraph_baselines/summarize_preliminary.py baseline/specgraph_baselines/results_bm25 baseline/specgraph_baselines/results_ablation_query_code baseline/specgraph_baselines/results_ablation_question
```

Replay evaluates Dense:BM25 ratios 1:1, 2:1 and 3:1, and verifies that 1:1 matches
the saved equal-weight clause ranking. Its evidence comes from the branch with
the larger weighted clause-rank contribution, with Dense breaking ties. This
differs from online chunk-RRF evidence selection and does not affect clause
metrics. Replayed evidence is not passed to a reranker.

## Preserved Files

| File or directory | Purpose |
| --- | --- |
| `retrieval_baselines.py` | Main retrieval, validation and evaluation CLI |
| `test_retrieval_baselines.py` | 18 regression checks |
| `run_preliminary.ps1` | Windows model-execution wrapper |
| `replay_fusion.py` | Fixed fusion-weight grid from saved rankings |
| `summarize_preliminary.py` | Validates common inputs and produces comparison tables |
| `PRELIMINARY_RESULTS.md` | Final English results report |
| `results_bm25/` | Original unfiltered BM25 reference |
| `results_preliminary_bge_small/` | Original model execution, including preprocessed BM25 and cross-encoder |
| `results_question_only/` | Actual question-only model execution |
| `results_ablation_query_code/` | Weight grid for question plus code |
| `results_ablation_question/` | Weight grid for question-only retrieval |
| `preliminary_summary/` | Comparison CSV/JSON and model commits |

Run directories contain rankings, metrics, configuration hashes, corpus mappings
and audits. Actual model runs also retain the exact executed source snapshots;
keep those snapshots for provenance and run the normal CLI for reproduction.
The environment and model cache live under `baseline/.venv/` and
`baseline/.cache/huggingface/`, excluded from version control.
