# SpecGraph-RAG Project

This repository contains the current N4950 corpus and benchmark, and the completed text retrieval baseline experiments.

## Start Here

- **Results and analysis:** [English preliminary results](baseline/specgraph_baselines/PRELIMINARY_RESULTS.md).
- **Run and reproduce:** [Baseline README](baseline/specgraph_baselines/README.md).
- **Machine-readable comparison:** [All 12 configurations](baseline/specgraph_baselines/preliminary_summary/comparison.csv).

The current preliminary references are question-only Dense retrieval at k=10
(74.15% recall, 57.14% complete coverage) and question-only 2:1 weighted RRF at
k=5 (65.82% recall, 48.98% complete coverage). These configurations were compared
on the available 49-query benchmark; the results are exploratory.

## Workspace Contents

| Path | Purpose |
| --- | --- |
| `data/processed/chunks.jsonl` | 454 specification chunks, representing 82 clauses |
| `data/benchmark/spec_benchmark_v1.jsonl` | 49 labeled retrieval queries |
| `data/graph/spec_graph.json` | Shared graph data for the broader project |
| `baseline/specgraph_baselines/` | Retrieval code, tests, report, and preserved experiment results |
| `baseline/.venv/` | Local CPU Python environment, excluded from version control |
| `baseline/.cache/huggingface/` | Cached model weights for offline runs, excluded from version control |

## Quick Run

Create the Python environment using the [baseline setup instructions](baseline/specgraph_baselines/README.md), then run from the project root:

```powershell
# Run the recommended question-only baselines and 2:1 fusion.
# A separate output directory preserves the recorded experiments.
& baseline/specgraph_baselines/run_preliminary.ps1 -SkipRerank -QueryMode question -DenseWeight 2 -OutputName results_reproduction
```

This produces BM25, Dense, and weighted Hybrid rankings together, allowing both
the k=10 Dense and k=5 Hybrid references to be evaluated in one run. Graph
retrieval, GNN training, and generation are outside this baseline implementation.
The first model run downloads weights; add `-Offline` on later runs once the
models have been cached. Local environments and model weights are not versioned.
