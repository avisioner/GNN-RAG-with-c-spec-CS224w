# SpecGraph-RAG

Retrieval experiments over the C++23 N4950 specification: BM25, BGE-small Dense,
weighted reciprocal rank fusion, optional cross-encoder reranking, and Dense +
GraphExp. Evaluation retrieves distinct clauses with representative chunk evidence.

**Start with [Results](results/README.md)** for the combined findings, detailed
results and saved runs. All result documents and experiment outputs live under
`results/`; code and reproduction instructions stay with their implementations.

| Selected result | Recall@10 | Complete coverage@10 |
| --- | ---: | ---: |
| Original question + code, Hybrid 2:1 | 70.24% | 51.02% |
| Same input + automatic operator glossary, Hybrid 2:1 | **91.16%** | **79.59% (39/49)** |
| Question-only Dense reference | 74.15% | 57.14% |

These are exploratory comparisons on the same 49 queries, not held-out test
estimates. The glossary preserves the original question and full code. It uses
operator spellings only; benchmark labels and manual rewrites are excluded.

## Repository

| Directory | Contents |
| --- | --- |
| [baseline/](baseline/specgraph_baselines/README.md) | Text retrieval code, environment setup and tests |
| [graph/](graph/specgraph_graph/README.md) | Clause graph construction and Dense + GraphExp |
| [experiments/](experiments/query_diagnostics/README.md) | Paired query representation experiment and verification |
| [results/](results/README.md) | All results, rankings, metrics, audits and source snapshots |
| `data/` | Canonical corpus (454 chunks, 82 clauses), 49-query benchmark and source graph |
| `artifacts/` | Reusable clause graph and full question-only clause scores |
| [docs/](docs/README.md) | Dataset review guidance and milestone writing |

## Reproduce

Follow the [baseline setup](baseline/specgraph_baselines/README.md), then run
from the project root:

```powershell
& baseline/specgraph_baselines/run_preliminary.ps1 -SkipRerank -QueryMode query_code -QueryTransform operator_gloss -DenseWeight 2 -EmbeddingRevision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a -OutputName operator_gloss_reproduction
```

Outputs go to `results/baseline/operator_gloss_reproduction/`. Add `-Offline`
when weights are cached. Python environments and downloaded weights remain
local and are excluded from Git. See [Results](results/README.md) for no-model
checks and the original question-only reference.

GNN training and answer generation are not implemented by these baselines.
