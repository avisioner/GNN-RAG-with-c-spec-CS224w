# Milestone progress: graph work (Aaron)

Last updated 2026-10-30. Milestone due 2026-11-10.

## Checklist

| Step | Status | Output |
|---|---|---|
| 0. Explore repo, plan | done | (this log) |
| 1. Data validation report | done | `results/milestone/data_validation.md` |
| 2. Clause graph + loader + stats | done | `artifacts/graph/clause_graph.json`, `clause_graph.load_clause_graph()` |
| 3. `graph_expand` (Dense + GraphExp) + variants + grid | done | `graph/specgraph_graph/graph_expand.py` |
| 4. Evaluation, CIs, paired, reachability | done | `results/milestone/graph_expansion_results.md`, `per_query.csv`, `run_config.json` |
| 5. Report section draft | done (for Aaron to edit) | `docs/milestone/graph_section_draft.md` |
| 6a. `uses_term` edges | not started (stretch, needs go-ahead) | |
| 6b. PyG export | not started (stretch, needs go-ahead) | |
| Git | **not committed**: Aaron makes all commits and pushes | |

Reproduce everything: `python graph/specgraph_graph/run_milestone.py` (numpy only), tests: `python -m pytest graph/specgraph_graph/tests -q` (25 pass).

## Decision log

- Naming: Erica's "Hybrid" = BM25 + Dense reciprocal-rank fusion. The project doc used "hybrid" for dense-then-graph-expansion. Ours is always **Dense + GraphExp**.
- Input mode: question-only, matching Erica's reported reference configuration. She chose that mode on the same 49 queries (exploratory), so it is not a held-out choice.
- Baseline rows use Erica's saved rankings (`results_question_only`, `results_ablation_question` for 2:1 RRF), evaluated with her `metrics_at_k`.
- Full clause scores: her CLI saves only top-50 clause lists, no scores. `clause_scores.py` recomputes them with her functions/model (BGE-small, same prefix); clause = max over chunks (her best-chunk rule). These reproduce her saved Dense and BM25 top-50 rankings for 49/49 queries, and `run_milestone.py` asserts that on every run.
- Environment: created `baseline/.venv` (gitignored, Erica's documented location) from her lock file, with macOS torch 2.15.0 instead of `+cpu`. Reproduced her question-only run: identical rankings for all three methods.
- Graph built from `chunks.jsonl` (canonical). `#` anchors stripped from reference targets. Out-of-corpus targets dropped (counts in stats). Self-loops removed (there are none in the data).
- Sibling edges added: 0 of 16 parents are corpus clauses, so `parent` is empty.
- Defaults fixed before results: seed_k=5, hops=1, λ=0.5, edges = references + parent + sibling. The grid is reported but not used for selection.
- Extra variants beyond the brief: `references_plain` (= spec_graph.json's edge set) and `references_general` (refs to chunk-less sections redirected to `<id>.general`). Both are exploratory rows only.
- Headline: coverage@10; @5 reported too (the doc is inconsistent).
- Ties: score desc, then clause ID asc (= Erica's convention).
- Result: Dense + GraphExp vs Dense on cov@10 is 0 wins / 3 losses / 46 ties (−6.1 pts, CI [−14.3, 0.0]); on cov@5, 2 wins / 0 losses. Within noise.

## Open questions

**For Monami**
- 8 `multi_hop` queries have 2 gold clauses (specqa-003, 004, 006, 011, 018, 020, 048, 049). Is the label intended (2 clauses that are far apart), or should they be `two_clause`? Not relabeled.
- Only 11 distinct gold clauses across 49 queries (`expr.post.incr` in 21). Will the 80–150 query expansion broaden coverage? This matters for the GNN, which could learn a clause prior.
- `spec_graph.json` drops 97 references with `#` anchors (`#nt:`, `#def:`); probably a builder bug. Fix upstream?
- OK to freeze a dev/test split of the benchmark for model selection in weeks 4–5?

**For Erica**
- Could the CLI optionally save full per-clause scores, not just top-50 ranks? We currently recompute them in a wrapper.
- Confirm "Hybrid" in the report means question-only 2:1 RRF (her @5 reference) or 1:1. Both are in our table.

## AI-use log

All of the following was written with Claude Code (Claude Opus 5.5) on 2026-10-30, directed and reviewed by Aaron. No teammate files were modified.

Created:
- `graph/specgraph_graph/common.py`, `validate_data.py`, `clause_graph.py`, `clause_scores.py`, `graph_expand.py`, `evaluate.py`, `run_milestone.py`, `README.md`
- `graph/specgraph_graph/tests/test_graph_milestone.py`
- `artifacts/graph/clause_graph.json`, `artifacts/scores/question_scores.json` (generated)
- `results/milestone/data_validation.md`, `graph_expansion_results.md`, `per_query.csv`, `run_config.json` (generated)
- `docs/milestone/graph_section_draft.md` (draft prose for Aaron to edit)
- `docs/claude/milestone_progress.md` (this file)

Local only, not versioned: `baseline/.venv/`, `baseline/.cache/huggingface/` (gitignored).
