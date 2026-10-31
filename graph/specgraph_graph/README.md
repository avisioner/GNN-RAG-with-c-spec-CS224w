# Clause graph and Dense + GraphExp (milestone)

Graph construction, the non-learned graph-expansion baseline, and its evaluation
against Erica's text baselines. Results: [graph_expansion_results.md](../../results/milestone/graph_expansion_results.md).
Data audit: [data_validation.md](../../results/milestone/data_validation.md).

## Run

From the project root, with any Python 3.10+ that has `numpy` (and `pytest` for the tests):

```bash
python graph/specgraph_graph/run_milestone.py          # validation + graph + all result tables
python -m pytest graph/specgraph_graph/tests -q        # hand-checked tests
```

`run_milestone.py` reads the cached full clause scores in `artifacts/scores/question_scores.json`
and Erica's saved rankings in `baseline/specgraph_baselines/results_*`, so it loads no model.
To regenerate the scores you need Erica's environment (`baseline/specgraph_baselines/requirements.txt`):

```bash
python graph/specgraph_graph/clause_scores.py
```

`run_milestone.py` refuses to run if the cached scores stop reproducing Erica's saved Dense/BM25 rankings.

## Use the graph

```python
import sys; sys.path.insert(0, "graph/specgraph_graph")
from clause_graph import load_clause_graph, node_ids, adjacency
from graph_expand import graph_expand

g = load_clause_graph()                    # artifacts/graph/clause_graph.json
ids = node_ids(g)                          # 82 clause IDs, sorted; index of every score vector
A = adjacency(g, ("references", "sibling"))
new_scores = graph_expand(dense_scores, g) # seed_k=5, hops=1, lam=0.5, refs+parent+sibling
```

## Files

| File | Purpose |
|---|---|
| `common.py` | Paths, `norm_id`, `normalize_difficulty`, loaders (reuses Erica's reader) |
| `validate_data.py` | Step 1 data audit -> `results/milestone/data_validation.md` |
| `clause_graph.py` | Step 2 graph build, `load_clause_graph`, `adjacency`, stats |
| `clause_scores.py` | Full 82-clause BM25/Dense scores via Erica's functions |
| `graph_expand.py` | Step 3 `graph_expand` |
| `evaluate.py` | Metrics (Erica's `metrics_at_k`), bootstrap CI, paired, reachability |
| `run_milestone.py` | Step 4: one command for all tables, `per_query.csv`, `run_config.json` |
| `tests/` | pytest |
