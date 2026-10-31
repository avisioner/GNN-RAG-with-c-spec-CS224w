# Graph-expansion results (milestone)

Generated 2026-10-31 00:05 UTC by `python graph/specgraph_graph/run_milestone.py`.

**Setup.** 49 benchmark queries, 82 clauses, question-only input (Erica's reference mode). BM25 / Dense / Hybrid rows are Erica's saved rankings (`baseline/specgraph_baselines/results_question_only`, `results_ablation_question`); all rows go through the same metric code (`retrieval_baselines.metrics_at_k`). Dense + GraphExp reranks Dense's full 82-clause scores (clause = max over chunks; these reproduce her saved Dense ranking exactly) with seed_k=5, hops=1, λ=0.5, edges = references + parent + sibling, fixed in advance. "Hybrid" here is Erica's BM25+Dense reciprocal-rank fusion. Graph expansion is a separate method, always called **Dense + GraphExp**. Coverage = complete coverage (all gold clauses in top-k). All numbers are percentages.

> **Headline metric:** the project doc says "coverage@5" in one place and "every gold clause in the top 10" in another, so both are reported. Coverage@10 is the headline.

## Main table

| Method | Cov@1 | Cov@3 | Cov@5 | **Cov@10** | R@1 | R@3 | R@5 | R@10 | P@1 | P@3 | P@5 | P@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 26.5 | 28.6 | 36.7 | **42.9** | 34.7 | 42.2 | 53.1 | 57.8 | 44.9 | 19.7 | 15.9 | 9.0 | 0.496 |
| Dense | 24.5 | 34.7 | 42.9 | **57.1** | 32.0 | 52.2 | 60.9 | 74.1 | 40.8 | 26.5 | 20.0 | 12.9 | 0.581 |
| Hybrid RRF 1:1 | 28.6 | 34.7 | 46.9 | **53.1** | 39.8 | 52.4 | 61.9 | 70.2 | 53.1 | 25.9 | 19.6 | 11.8 | 0.599 |
| Hybrid RRF 2:1 | 28.6 | 34.7 | 49.0 | **51.0** | 39.8 | 53.6 | 65.8 | 69.9 | 53.1 | 27.2 | 21.6 | 11.8 | 0.601 |
| Dense + GraphExp | 24.5 | 32.7 | 46.9 | **51.0** | 29.9 | 50.2 | 61.7 | 69.4 | 36.7 | 24.5 | 20.0 | 11.8 | 0.553 |

## By difficulty label

| Method | single_clause Cov@5 | single_clause Cov@10 | single_clause R@5 | single_clause R@10 | two_clause Cov@5 | two_clause Cov@10 | two_clause R@5 | two_clause R@10 | multi_hop Cov@5 | multi_hop Cov@10 | multi_hop R@5 | multi_hop R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 100.0 | 100.0 | 100.0 | 100.0 | 11.8 | 29.4 | 41.2 | 50.0 | 5.9 | 5.9 | 23.5 | 28.4 |
| Dense | 100.0 | 100.0 | 100.0 | 100.0 | 29.4 | 64.7 | 52.9 | 76.5 | 5.9 | 11.8 | 34.3 | 49.0 |
| Hybrid RRF 1:1 | 100.0 | 100.0 | 100.0 | 100.0 | 35.3 | 52.9 | 55.9 | 70.6 | 11.8 | 11.8 | 34.3 | 43.6 |
| Hybrid RRF 2:1 | 100.0 | 100.0 | 100.0 | 100.0 | 41.2 | 47.1 | 61.8 | 67.6 | 11.8 | 11.8 | 39.7 | 45.6 |
| Dense + GraphExp | 100.0 | 100.0 | 100.0 | 100.0 | 41.2 | 52.9 | 58.8 | 67.6 | 5.9 | 5.9 | 30.9 | 44.1 |

n = {'single_clause': 15, 'two_clause': 17, 'multi_hop': 17}. Eight `multi_hop` queries have only 2 gold clauses (see data_validation.md), so the table below groups by gold-set size as well.

| Method | 1 gold Cov@10 | 1 gold R@10 | 2 gold Cov@10 | 2 gold R@10 | 3 gold Cov@10 | 3 gold R@10 | 4 gold Cov@10 | 4 gold R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 100.0 | 100.0 | 24.0 | 44.0 | 0.0 | 29.2 | 0.0 | 0.0 |
| Dense | 100.0 | 100.0 | 52.0 | 66.0 | 0.0 | 54.2 | 0.0 | 50.0 |
| Hybrid RRF 1:1 | 100.0 | 100.0 | 44.0 | 62.0 | 0.0 | 45.8 | 0.0 | 25.0 |
| Hybrid RRF 2:1 | 100.0 | 100.0 | 40.0 | 60.0 | 0.0 | 50.0 | 0.0 | 25.0 |
| Dense + GraphExp | 100.0 | 100.0 | 40.0 | 58.0 | 0.0 | 50.0 | 0.0 | 50.0 |

n = {1: 15, 2: 25, 3: 8, 4: 1}.

## Uncertainty (95% bootstrap CI, 2000 query resamples, seed 0)

| Method | Cov@10 | R@5 | multi_hop Cov@10 |
|---|---:|---:|---:|
| BM25 | 42.9 [30.6, 57.1] | 53.1 [41.8, 64.3] | 5.9 [0.0, 17.6] |
| Dense | 57.1 [42.9, 71.4] | 60.9 [50.3, 70.7] | 11.8 [0.0, 29.4] |
| Hybrid RRF 1:1 | 53.1 [38.8, 67.3] | 61.9 [50.3, 72.8] | 11.8 [0.0, 29.4] |
| Hybrid RRF 2:1 | 51.0 [36.7, 65.3] | 65.8 [55.3, 75.9] | 11.8 [0.0, 29.4] |
| Dense + GraphExp | 51.0 [36.7, 65.3] | 61.7 [50.5, 72.1] | 5.9 [0.0, 17.6] |

**Paired: Dense + GraphExp vs Dense**, per query (wins / losses / ties, and the bootstrap CI of the mean difference in points):

| Subset | Metric | Wins | Losses | Ties | Mean diff [95% CI] |
|---|---|---:|---:|---:|---:|
| all (n=49) | complete_coverage@10 | 0 | 3 | 46 | -6.1 [-14.3, +0.0] |
| all (n=49) | complete_coverage@5 | 2 | 0 | 47 | +4.1 [+0.0, +10.2] |
| all (n=49) | recall@5 | 2 | 2 | 45 | +0.9 [-2.0, +4.4] |
| all (n=49) | recall@10 | 1 | 6 | 42 | -4.8 [-9.5, -0.7] |
| multi_hop (n=17) | complete_coverage@10 | 0 | 1 | 16 | -5.9 [-17.6, +0.0] |
| multi_hop (n=17) | complete_coverage@5 | 0 | 0 | 17 | +0.0 [+0.0, +0.0] |
| multi_hop (n=17) | recall@5 | 0 | 2 | 15 | -3.4 [-8.8, +0.0] |
| multi_hop (n=17) | recall@10 | 1 | 3 | 13 | -4.9 [-13.7, +3.9] |

## Variants (exploratory)

| Variant | Cov@5 | Cov@10 | R@5 | R@10 | multi_hop Cov@10 | multi_hop R@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 42.9 | 57.1 | 60.9 | 74.1 | 11.8 | 49.0 | 0.581 |
| Dense + GraphExp (default: refs+parent+sibling, 1 hop) | 46.9 | 51.0 | 61.7 | 69.4 | 5.9 | 44.1 | 0.553 |
| Dense + GraphExp, hops = 2 | 44.9 | 51.0 | 61.4 | 69.0 | 5.9 | 43.1 | 0.547 |
| Dense + GraphExp, references only | 46.9 | 51.0 | 62.1 | 70.1 | 5.9 | 46.1 | 0.559 |
| Dense + GraphExp, sibling only | 42.9 | 51.0 | 60.2 | 70.7 | 5.9 | 42.2 | 0.498 |
| Dense + GraphExp, parent only (no edges on this data = Dense) | 42.9 | 57.1 | 60.9 | 74.1 | 11.8 | 49.0 | 0.581 |
| Dense + GraphExp, references_plain only (= spec_graph.json refs) | 44.9 | 55.1 | 61.1 | 73.5 | 11.8 | 52.9 | 0.572 |
| Dense + GraphExp, refs + references_general + sibling | 46.9 | 51.0 | 62.2 | 70.4 | 5.9 | 44.1 | 0.553 |
| Hybrid 2:1 + GraphExp (hybrid seeds) | 46.9 | 53.1 | 65.5 | 70.9 | 11.8 | 45.6 | 0.582 |

`references_plain` is the reference edge set of `data/graph/spec_graph.json` (it drops the 97 `#`-anchored references). `references_general` redirects references to chunk-less parent sections (e.g. `over.oper`) to that section's `.general` clause.

## Sensitivity grid (exploratory, Dense + GraphExp, default edges, seed_k=5)

Not used to choose a setting: choosing from this grid would be tuning on the test set. Model selection needs a dev/test split.

| λ | hops | Cov@5 | Cov@10 | R@5 | R@10 | multi_hop Cov@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 1 | 46.9 | 53.1 | 62.9 | 71.4 | 5.9 |
| 0.25 | 2 | 44.9 | 53.1 | 61.9 | 70.7 | 5.9 |
| 0.5 | 1 | 46.9 | 51.0 | 61.7 | 69.4 | 5.9 |
| 0.5 | 2 | 44.9 | 51.0 | 61.4 | 69.0 | 5.9 |
| 1.0 | 1 | 44.9 | 53.1 | 61.4 | 70.1 | 5.9 |
| 1.0 | 2 | 44.9 | 51.0 | 60.7 | 69.0 | 5.9 |

## Reachability from the Dense top-5 (1 hop)

*Gold in reach*: share of a query's gold clauses in seeds ∪ N(seeds); *chance*: share of all 82 clauses in seeds ∪ N(seeds). *Missed gold found*: of the gold clauses Dense did not put in its top 5, the share that are graph neighbors of the seeds; *chance*: share of all non-seed clauses that are neighbors. Macro over queries (missed-gold columns only over queries with a missed gold clause).

| Edges | Difficulty | n | Gold in reach | Chance | Missed gold found | Chance (non-seed) | n w/ missed |
|---|---|---:|---:|---:|---:|---:|---:|
| refs+parent+sibling | all | 49 | 94.9 | 59.0 | 91.1 | 55.8 | 28 |
| refs+parent+sibling | single_clause | 15 | 100.0 | 58.7 | n/a | n/a | 0 |
| refs+parent+sibling | two_clause | 17 | 91.2 | 61.0 | 87.5 | 57.7 | 12 |
| refs+parent+sibling | multi_hop | 17 | 94.1 | 57.2 | 93.8 | 54.5 | 16 |
| references | all | 49 | 91.8 | 44.2 | 85.7 | 40.5 | 28 |
| references | single_clause | 15 | 100.0 | 43.2 | n/a | n/a | 0 |
| references | two_clause | 17 | 85.3 | 45.8 | 79.2 | 40.8 | 12 |
| references | multi_hop | 17 | 91.2 | 43.5 | 90.6 | 40.3 | 16 |
| sibling | all | 49 | 74.8 | 33.7 | 35.7 | 30.5 | 28 |
| sibling | single_clause | 15 | 100.0 | 32.8 | n/a | n/a | 0 |
| sibling | two_clause | 17 | 70.6 | 33.5 | 37.5 | 31.0 | 12 |
| sibling | multi_hop | 17 | 56.9 | 34.6 | 34.4 | 30.2 | 16 |

