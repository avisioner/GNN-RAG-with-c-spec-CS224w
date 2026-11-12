# GNN milestone progress (Aaron)

Read this first in any new session. Last updated 2026-11-11.

Reproduce: `baseline/.venv/bin/python graph/specgraph_gnn/run_gnn.py` (105 runs, ~40 s on 6 workers; cached in
`results/gnn_milestone/cache/`, gitignored). Tests: `baseline/.venv/bin/python -m pytest graph/specgraph_gnn/tests
graph/specgraph_graph/tests -q` (37 pass).

## Checklist

| Step | Status | Output |
|---|---|---|
| 0. Explore, plan; extract old corpus | done | `artifacts/corpus_v1/chunks.jsonl` (sha256 checked on load) |
| 1. Score benchmark with Erica's functions (+ mean-chunk dense) | done; matches `question_scores.json` | `scoring.py` |
| 2. Features, models, loss, tests | done (12 tests) | `features.py`, `models.py`, `train.py`, `tests/test_gnn.py` |
| 3. 5-fold CV on real queries, popularity prior, 7 models (incl. no-sibling R-GCN/SAGE/GAT) | done | `run_gnn.py`, `results/gnn_milestone/RESULTS.md`, `per_query.csv`, `run_config.json` |
| 4. Report draft | done (for Aaron to edit) | `docs/milestone/gnn_section_draft.md` |
| 5. Stretch (uses_term edges, 460-clause corpus) | not started; needs go-ahead | |
| Git | **not committed**: Aaron commits and pushes | |

## Decision log

- **Real data only** (Aaron, as agreed in the team meeting): no generated queries.
- Protocol: 5-fold CV over the 49 queries, stratified by difficulty (seed 0). For test fold f: train on 3 folds, early-stop on fold (f+1) mod 5, test on f. With about 29 training queries and batch 64, one epoch is one gradient step. The brief's hyperparameters are unchanged and nothing was tuned on results.
- Clause-popularity prior (kept at Aaron's request): rank by gold frequency in the 4 non-test folds, ties by dense score. It is a check, not a model.
- Old 82-clause corpus from `7a29ead` (byte-identical to `eb7c8b9`). Erica's baselines are not rerun. Her saved rankings are re-evaluated and must match `results/milestone/per_query.csv` (asserted). The benchmark scores via her functions must match `question_scores.json` (asserted).
- Relations: 3, per Aaron: `references` forward, `references` reversed, `sibling` (undirected, both directions in one relation). The no-sibling ablation uses 2 and is run for R-GCN, GraphSAGE and GAT (Aaron), to test whether typed relations make the model robust to the weaker sibling edges. `parent` has 0 edges. No edge type was renamed, merged or added.
- Features: 7 (constant-1 slot dropped; layers have biases). Degree uses the model's own edge types. The MLP gets the refs+sibling degree.
- MLP control: same depth and width, with Linear layers in place of convolutions. GAT: 4 heads averaged. SAGE: mean.
- Paired comparisons: W/L/T listed per seed; CI on the seed-averaged per-query difference.
- The seen/unseen *clause* split is degenerate (49/49 seen), so RESULTS.md splits by seen/unseen gold **set** (28/21) and reports Popular@10.

## Key results (coverage@10, 49 queries)

Dense 57.1. Dense + GraphExp 51.0. MLP 55.1 ± 1.7. **R-GCN 73.5 ± 1.7** (vs Dense +16.3, CI [+4.1, +29.3]; multi-hop +47).
Sibling ablation (with − without): R-GCN −0.7 [−7.5, +5.4] (n.s.); GraphSAGE −7.5 [−12.2, −3.4] (54.4 vs 61.9; multi-hop −15.7); GAT −4.1 [−9.5, +0.7] (52.4 vs 56.5). Siblings make up 762 of 1,440 directed edges. Mixed into one untyped neighbor average they dilute the reference signal; the R-GCN gives them their own weight matrix and is unaffected. Even without siblings, the R-GCN (74.1) beats GraphSAGE (61.9), so the separate forward/reverse reference relations also matter. **Popularity prior 91.8.** Popular@10 is 70% for R-GCN, 41% for Dense and 49% for MLP. The most likely explanation is that the R-GCN learns the popularity prior through structural node identity.

## Open questions

**For Monami**
- The new queries need many more distinct gold clauses. Currently there are 11 across 49 queries.
- The 8 two-gold `multi_hop` queries (from the Oct 30 log) are still unresolved.
- The meeting mentioned ~89 queries. Only 49 are in the repo.

**For Erica**
- Is "Hybrid" in the report RRF 2:1 or 1:1? Both are in the tables.

**For Aaron**
- Should we run a rewired-graph control (degree-preserving edge shuffle)? It takes about 30 s. Not run; needs your OK.

## Draft note to Monami (not sent)

> Hi Monami, a few things I hit while doing the GNN today:
> - `3f4771a` replaced `data/processed/chunks.jsonl` in place with the 460-clause parse, so the existing baseline and graph results no longer reproduce from `main`. Also, `run_milestone.py` would now rebuild `clause_graph.json` from the new file. I used the old corpus from `7a29ead`, saved as `artifacts/corpus_v1/chunks.jsonl`.
> - `data/graph/spec_graph.json` hasn't been rebuilt for the 460 clauses.
> - Only 49 queries are in the repo. Are the ~89 somewhere else?
> - The biggest issue for the GNN: the 49 queries have only 11 distinct gold clauses (`expr.post.incr` is in 21). A "rank by popularity" baseline gets 91.8% coverage@10, better than any model. It would help a lot if the new queries spread gold over many more clauses.

## AI-use log

All of the following was written with Claude Code (Claude Opus 5.5) on 2026-11-11, directed and reviewed by Aaron. No teammate files were modified.

Created:
- `artifacts/corpus_v1/chunks.jsonl` (verbatim `git show 7a29ead:data/processed/chunks.jsonl`)
- `graph/specgraph_gnn/gnn_common.py`, `scoring.py`, `features.py`, `models.py`, `train.py`, `run_gnn.py`, `tests/test_gnn.py`
- `results/gnn_milestone/RESULTS.md`, `per_query.csv`, `run_config.json` (generated), `.gitignore` (ignores `cache/`)
- `docs/milestone/gnn_section_draft.md`, `docs/milestone/GNN_PROGRESS.md` (this file)

Environment (local, not versioned): installed `torch_geometric` 2.8.1 into `baseline/.venv` with `uv pip`. It is not in Erica's lock file.
