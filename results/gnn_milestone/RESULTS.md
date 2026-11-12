# GNN reranker: milestone results

Generated 2026-11-12 01:55 UTC by `graph/specgraph_gnn/run_gnn.py`. Old 82-clause corpus (`artifacts/corpus_v1/chunks.jsonl`, = commit 7a29ead), 49 benchmark queries, question-only input. Metrics from Erica's `metrics_at_k`. Rankings over all 82 clauses, ties by clause ID.

- **Real data only.** GNN/MLP rows: 5-fold cross-validation over the 49 benchmark queries, stratified by difficulty (fold sizes [9, 10, 10, 10, 10]). Per test fold: 3 folds train, 1 fold for early stopping, 1 fold test. Every query is scored once by a model that never saw it.
- Model rows: mean ± std over seeds [0, 1, 2]. Baselines are deterministic and use no training.
- **Clause-popularity prior** is not a model: it ranks clauses by how often they are gold in the other 4 folds, ignoring the question (dense score only breaks ties). It measures how far one can get by memorizing which clauses are popular.
- Headline metric: **coverage@10** (all gold clauses in the top 10). With 17 two-clause and 17 multi-hop queries, one query moves a per-difficulty number by ~5.9 points.
- Single-clause queries are at 100% coverage@5 for every text baseline, so all differences come from two-clause and multi-hop queries.

## Main table

| Method | Cov@1 | Cov@3 | Cov@5 | **Cov@10** | R@1 | R@3 | R@5 | R@10 | P@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 26.5 | 28.6 | 36.7 | **42.9** | 34.7 | 42.2 | 53.1 | 57.8 | 15.9 | 0.496 |
| Dense | 24.5 | 34.7 | 42.9 | **57.1** | 32.0 | 52.2 | 60.9 | 74.1 | 20.0 | 0.581 |
| Hybrid RRF 1:1 | 28.6 | 34.7 | 46.9 | **53.1** | 39.8 | 52.4 | 61.9 | 70.2 | 19.6 | 0.599 |
| Hybrid RRF 2:1 | 28.6 | 34.7 | 49.0 | **51.0** | 39.8 | 53.6 | 65.8 | 69.9 | 21.6 | 0.601 |
| Dense + GraphExp | 24.5 | 32.7 | 46.9 | **51.0** | 29.9 | 50.2 | 61.7 | 69.4 | 20.0 | 0.553 |
| Clause-popularity prior | 4.1 | 20.4 | 40.8 | **91.8** | 20.2 | 44.6 | 64.1 | 94.2 | 25.7 | 0.627 |
| MLP (no graph) | 29.9 ± 1.0 | 35.4 ± 1.0 | 43.5 ± 1.0 | **55.1 ± 1.7** | 41.5 ± 0.8 | 52.7 ± 1.9 | 60.4 ± 1.2 | 71.9 ± 1.6 | 19.5 ± 0.5 | 0.615 |
| R-GCN | 27.9 ± 1.0 | 38.8 ± 0.0 | 49.0 ± 4.4 | **73.5 ± 1.7** | 42.6 ± 2.3 | 60.9 ± 0.4 | 72.6 ± 2.1 | 87.0 ± 0.7 | 25.2 ± 0.8 | 0.716 |
| R-GCN (no sibling) | 27.2 ± 1.0 | 38.1 ± 1.9 | 49.0 ± 7.3 | **74.1 ± 7.9** | 42.1 ± 1.6 | 61.9 ± 2.0 | 70.9 ± 5.4 | 87.5 ± 4.0 | 24.5 ± 2.2 | 0.718 |
| GraphSAGE | 29.9 ± 1.0 | 34.0 ± 1.0 | 44.2 ± 3.5 | **54.4 ± 1.9** | 42.0 ± 0.7 | 53.6 ± 1.4 | 61.8 ± 3.7 | 72.7 ± 3.5 | 20.3 ± 1.7 | 0.623 |
| GraphSAGE (no sibling) | 29.3 ± 1.0 | 35.4 ± 1.0 | 48.3 ± 3.5 | **61.9 ± 2.5** | 43.6 ± 2.3 | 56.5 ± 2.2 | 65.3 ± 2.9 | 78.6 ± 1.3 | 21.8 ± 1.4 | 0.671 |
| GAT | 26.5 ± 5.8 | 32.7 ± 1.7 | 42.9 ± 1.7 | **52.4 ± 3.8** | 38.9 ± 5.6 | 52.4 ± 0.3 | 61.7 ± 1.2 | 71.8 ± 1.8 | 20.3 ± 0.4 | 0.609 |
| GAT (no sibling) | 29.3 ± 1.0 | 34.7 ± 1.7 | 44.9 ± 1.7 | **56.5 ± 2.5** | 41.1 ± 0.8 | 54.0 ± 0.6 | 63.1 ± 1.9 | 74.6 ± 1.5 | 20.8 ± 1.0 | 0.632 |

## By difficulty (Cov@5 / Cov@10 / R@5 / R@10)

| Method | single_clause Cov@5 | single_clause Cov@10 | single_clause R@5 | single_clause R@10 | two_clause Cov@5 | two_clause Cov@10 | two_clause R@5 | two_clause R@10 | multi_hop Cov@5 | multi_hop Cov@10 | multi_hop R@5 | multi_hop R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 100.0 | 100.0 | 100.0 | 100.0 | 11.8 | 29.4 | 41.2 | 50.0 | 5.9 | 5.9 | 23.5 | 28.4 |
| Dense | 100.0 | 100.0 | 100.0 | 100.0 | 29.4 | 64.7 | 52.9 | 76.5 | 5.9 | 11.8 | 34.3 | 49.0 |
| Hybrid RRF 1:1 | 100.0 | 100.0 | 100.0 | 100.0 | 35.3 | 52.9 | 55.9 | 70.6 | 11.8 | 11.8 | 34.3 | 43.6 |
| Hybrid RRF 2:1 | 100.0 | 100.0 | 100.0 | 100.0 | 41.2 | 47.1 | 61.8 | 67.6 | 11.8 | 11.8 | 39.7 | 45.6 |
| Dense + GraphExp | 100.0 | 100.0 | 100.0 | 100.0 | 41.2 | 52.9 | 58.8 | 67.6 | 5.9 | 5.9 | 30.9 | 44.1 |
| Clause-popularity prior | 46.7 | 86.7 | 46.7 | 86.7 | 52.9 | 100.0 | 73.5 | 100.0 | 23.5 | 88.2 | 70.1 | 95.1 |
| MLP (no graph) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 31.4 ± 2.8 | 54.9 ± 2.8 | 52.9 ± 2.4 | 68.6 ± 1.4 | 5.9 ± 0.0 | 15.7 ± 5.5 | 33.0 ± 1.2 | 50.3 ± 3.2 |
| R-GCN | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 33.3 ± 7.3 | 64.7 ± 4.8 | 60.8 ± 1.4 | 79.4 ± 2.4 | 19.6 ± 5.5 | 58.8 ± 0.0 | 60.1 ± 4.7 | 83.0 ± 1.4 |
| R-GCN (no sibling) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 37.3 ± 11.1 | 76.5 ± 8.3 | 62.7 ± 8.4 | 87.3 ± 5.5 | 15.7 ± 10.0 | 49.0 ± 14.7 | 53.4 ± 7.1 | 76.6 ± 6.0 |
| GraphSAGE | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 31.4 ± 7.3 | 54.9 ± 2.8 | 52.9 ± 7.2 | 72.5 ± 5.0 | 7.8 ± 2.8 | 13.7 ± 2.8 | 37.1 ± 3.7 | 48.7 ± 5.1 |
| GraphSAGE (no sibling) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 35.3 ± 4.8 | 60.8 ± 2.8 | 57.8 ± 3.7 | 77.5 ± 1.4 | 15.7 ± 5.5 | 29.4 ± 9.6 | 42.2 ± 5.2 | 60.9 ± 5.0 |
| GAT | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 27.5 ± 7.3 | 47.1 ± 0.0 | 52.9 ± 4.2 | 66.7 ± 1.4 | 7.8 ± 5.5 | 15.7 ± 11.1 | 36.8 ± 3.9 | 52.1 ± 5.7 |
| GAT (no sibling) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 29.4 ± 4.8 | 54.9 ± 2.8 | 51.0 ± 3.7 | 72.5 ± 1.4 | 11.8 ± 0.0 | 19.6 ± 5.5 | 42.6 ± 4.2 | 54.2 ± 3.2 |

## Uncertainty (95% bootstrap CI over queries, 2000 resamples, seed 0; model rows use seed-averaged per-query values)

| Method | Cov@10 | R@5 | multi_hop Cov@10 |
|---|---:|---:|---:|
| BM25 | 42.9 [30.6, 57.1] | 53.1 [41.8, 64.3] | 5.9 [0.0, 17.6] |
| Dense | 57.1 [42.9, 71.4] | 60.9 [50.3, 70.7] | 11.8 [0.0, 29.4] |
| Hybrid RRF 1:1 | 53.1 [38.8, 67.3] | 61.9 [50.3, 72.8] | 11.8 [0.0, 29.4] |
| Hybrid RRF 2:1 | 51.0 [36.7, 65.3] | 65.8 [55.3, 75.9] | 11.8 [0.0, 29.4] |
| Dense + GraphExp | 51.0 [36.7, 65.3] | 61.7 [50.5, 72.1] | 5.9 [0.0, 17.6] |
| Clause-popularity prior | 91.8 [83.7, 98.0] | 64.1 [53.4, 73.8] | 88.2 [70.6, 100.0] |
| MLP (no graph) | 55.1 [42.2, 68.0] | 60.4 [49.9, 70.7] | 15.7 [3.9, 31.4] |
| R-GCN | 73.5 [61.9, 83.7] | 72.6 [65.0, 79.6] | 58.8 [37.3, 78.4] |
| R-GCN (no sibling) | 74.1 [63.3, 84.4] | 70.9 [63.0, 78.3] | 49.0 [29.4, 66.7] |
| GraphSAGE | 54.4 [41.5, 67.4] | 61.8 [51.9, 71.5] | 13.7 [2.0, 27.5] |
| GraphSAGE (no sibling) | 61.9 [49.7, 74.1] | 65.3 [55.4, 74.5] | 29.4 [13.7, 47.1] |
| GAT | 52.4 [40.1, 64.6] | 61.7 [51.6, 71.3] | 15.7 [5.9, 27.5] |
| GAT (no sibling) | 56.5 [43.5, 68.7] | 63.1 [53.3, 72.2] | 19.6 [5.9, 35.3] |

## Paired: R-GCN vs others (per query; wins/losses/ties per seed, CI on the seed-averaged difference)

| Comparison | Metric | Subset | W/L/T per seed | Mean diff [95% CI] |
|---|---|---|---|---:|
| R-GCN vs Dense | Cov@10 | all | 10/3/36, 11/2/36, 10/2/37 | +16.3 [+4.1, +29.3] |
| R-GCN vs Dense | Cov@10 | multi_hop | 8/0/9, 8/0/9, 8/0/9 | +47.1 [+25.5, +68.6] |
| R-GCN vs Dense | recall@5 | all | 15/5/29, 15/3/31, 15/2/32 | +11.7 [+4.6, +18.6] |
| R-GCN vs Dense | recall@5 | multi_hop | 9/1/7, 10/0/7, 10/0/7 | +25.8 [+15.0, +37.3] |
| R-GCN vs Dense + GraphExp | Cov@10 | all | 11/1/37, 13/1/35, 12/1/36 | +22.4 [+10.9, +34.0] |
| R-GCN vs Dense + GraphExp | Cov@10 | multi_hop | 9/0/8, 9/0/8, 9/0/8 | +52.9 [+33.3, +74.5] |
| R-GCN vs Dense + GraphExp | recall@5 | all | 14/5/30, 15/4/30, 16/3/30 | +10.8 [+3.7, +17.9] |
| R-GCN vs Dense + GraphExp | recall@5 | multi_hop | 10/1/6, 11/0/6, 12/0/5 | +29.2 [+18.5, +40.0] |
| R-GCN vs Hybrid RRF 2:1 | Cov@10 | all | 11/1/37, 13/1/35, 12/1/36 | +22.4 [+11.6, +34.7] |
| R-GCN vs Hybrid RRF 2:1 | Cov@10 | multi_hop | 8/0/9, 8/0/9, 8/0/9 | +47.1 [+25.5, +68.6] |
| R-GCN vs Hybrid RRF 2:1 | recall@5 | all | 11/6/32, 13/5/31, 13/4/32 | +6.7 [-0.3, +13.3] |
| R-GCN vs Hybrid RRF 2:1 | recall@5 | multi_hop | 8/2/7, 8/0/9, 9/0/8 | +20.4 [+9.6, +32.4] |
| R-GCN vs Clause-popularity prior | Cov@10 | all | 4/14/31, 4/12/33, 4/13/32 | -18.4 [-33.3, -4.1] |
| R-GCN vs Clause-popularity prior | Cov@10 | multi_hop | 2/7/8, 2/7/8, 2/7/8 | -29.4 [-58.8, -0.0] |
| R-GCN vs Clause-popularity prior | recall@5 | all | 12/12/25, 14/12/23, 14/13/22 | +8.4 [-5.2, +23.3] |
| R-GCN vs Clause-popularity prior | recall@5 | multi_hop | 1/5/11, 2/5/10, 2/6/9 | -10.0 [-23.0, +3.3] |
| R-GCN vs MLP (no graph) | Cov@10 | all | 9/1/39, 11/0/38, 9/1/39 | +18.4 [+9.5, +27.9] |
| R-GCN vs MLP (no graph) | Cov@10 | multi_hop | 8/0/9, 8/0/9, 6/0/11 | +43.1 [+25.5, +60.8] |
| R-GCN vs MLP (no graph) | recall@5 | all | 13/3/33, 17/5/27, 16/4/29 | +12.1 [+5.1, +18.9] |
| R-GCN vs MLP (no graph) | recall@5 | multi_hop | 8/1/8, 11/1/5, 11/1/5 | +27.1 [+14.7, +38.9] |
| R-GCN vs R-GCN (no sibling) | Cov@10 | all | 1/6/42, 2/3/44, 7/2/40 | -0.7 [-7.5, +5.4] |
| R-GCN vs R-GCN (no sibling) | Cov@10 | multi_hop | 1/2/14, 2/1/14, 6/1/10 | +9.8 [-2.0, +21.6] |
| R-GCN vs R-GCN (no sibling) | recall@5 | all | 3/10/36, 8/5/36, 11/3/35 | +1.6 [-2.3, +5.7] |
| R-GCN vs R-GCN (no sibling) | recall@5 | multi_hop | 2/4/11, 5/2/10, 7/2/8 | +6.7 [-0.2, +14.1] |

## Sibling ablation by architecture

Same model with vs without sibling edges (refs-only graph; the degree feature also drops siblings). Diff = with − without, seed-averaged per query, 95% bootstrap CI. W/L/T per seed pairs seed s with seed s.

| Architecture | Metric | Subset | With sibling | Without sibling | W/L/T per seed | Diff [95% CI] |
|---|---|---|---:|---:|---|---:|
| R-GCN | Cov@10 | all | 73.5 ± 1.7 | 74.1 ± 7.9 | 1/6/42, 2/3/44, 7/2/40 | -0.7 [-7.5, +5.4] |
| R-GCN | Cov@10 | multi_hop | 58.8 ± 0.0 | 49.0 ± 14.7 | 1/2/14, 2/1/14, 6/1/10 | +9.8 [-2.0, +21.6] |
| R-GCN | recall@5 | all | 72.6 ± 2.1 | 70.9 ± 5.4 | 3/10/36, 8/5/36, 11/3/35 | +1.6 [-2.3, +5.7] |
| R-GCN | recall@5 | multi_hop | 60.1 ± 4.7 | 53.4 ± 7.1 | 2/4/11, 5/2/10, 7/2/8 | +6.7 [-0.2, +14.1] |
| GraphSAGE | Cov@10 | all | 54.4 ± 1.9 | 61.9 ± 2.5 | 0/3/46, 0/6/43, 0/2/47 | -7.5 [-12.2, -3.4] |
| GraphSAGE | Cov@10 | multi_hop | 13.7 ± 2.8 | 29.4 ± 9.6 | 0/1/16, 0/5/12, 0/2/15 | -15.7 [-27.5, -5.9] |
| GraphSAGE | recall@5 | all | 61.8 ± 3.7 | 65.3 ± 2.9 | 6/2/41, 2/13/34, 4/5/40 | -3.5 [-7.1, +0.2] |
| GraphSAGE | recall@5 | multi_hop | 37.1 ± 3.7 | 42.2 ± 5.2 | 2/1/14, 1/7/9, 2/1/14 | -5.1 [-11.6, +1.8] |
| GAT | Cov@10 | all | 52.4 ± 3.8 | 56.5 ± 2.5 | 1/6/42, 4/3/42, 1/3/45 | -4.1 [-9.5, +0.7] |
| GAT | Cov@10 | multi_hop | 15.7 ± 11.1 | 19.6 ± 5.5 | 0/4/13, 2/0/15, 1/1/15 | -3.9 [-15.7, +5.9] |
| GAT | recall@5 | all | 61.7 ± 1.2 | 63.1 ± 1.9 | 2/6/41, 5/7/37, 3/0/46 | -1.4 [-4.4, +1.8] |
| GAT | recall@5 | multi_hop | 36.8 ± 3.9 | 42.6 ± 4.2 | 0/4/13, 2/4/11, 1/0/16 | -5.9 [-11.8, -0.3] |

## Benchmark issue: clause popularity

The 49 queries use only 11 distinct gold clauses (`expr.post.incr` is gold in 21). 49/49 test queries have every gold clause already gold in another fold, so a split by unseen *clauses* is empty. The table splits instead by whether the query's exact gold **set** occurs in another fold (28 seen, 21 unseen). *Popular@10* = mean share of the top 10 taken by clauses that are gold in another fold.

| Method | Cov@10 all | Cov@10 seen set (n=28) | Cov@10 unseen set (n=21) | R@5 seen set | R@5 unseen set | Popular@10 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 57.1 | 64.3 | 47.6 | 61.3 | 60.3 | 41% |
| Dense + GraphExp | 51.0 | 57.1 | 42.9 | 63.1 | 59.9 | 42% |
| Hybrid RRF 2:1 | 51.0 | 53.6 | 47.6 | 66.7 | 64.7 | 41% |
| Clause-popularity prior | 91.8 | 96.4 | 85.7 | 69.0 | 57.5 | 100% |
| MLP (no graph) | 55.1 ± 1.7 | 61.9 ± 1.7 | 46.0 ± 4.5 | 65.5 ± 2.2 | 53.7 ± 0.2 | 49% |
| R-GCN | 73.5 ± 1.7 | 67.9 ± 2.9 | 81.0 ± 0.0 | 73.6 ± 0.6 | 71.2 ± 5.7 | 70% |
| R-GCN (no sibling) | 74.1 ± 7.9 | 75.0 ± 5.1 | 73.0 ± 11.9 | 73.6 ± 4.9 | 67.3 ± 6.1 | 63% |
| GraphSAGE | 54.4 ± 1.9 | 61.9 ± 1.7 | 44.4 ± 2.2 | 66.7 ± 4.5 | 55.4 ± 2.8 | 49% |
| GraphSAGE (no sibling) | 61.9 ± 2.5 | 67.9 ± 2.9 | 54.0 ± 9.8 | 67.3 ± 3.0 | 62.7 ± 3.8 | 52% |
| GAT | 52.4 ± 3.8 | 59.5 ± 4.5 | 42.9 ± 3.9 | 65.1 ± 2.2 | 57.3 ± 1.3 | 50% |
| GAT (no sibling) | 56.5 ± 2.5 | 61.9 ± 4.5 | 49.2 ± 2.2 | 64.1 ± 1.8 | 61.8 ± 3.0 | 50% |

## Early stopping

Validation fold = the fold after the test fold. Max 100 epochs, patience 15. With about 29 training queries and batch size 64, one epoch is one gradient step.

| Model | Val-fold R@5 at best epoch | Test R@5 | Best epoch (mean, range) |
|---|---:|---:|---:|
| MLP (no graph) | 66.6 ± 5.4 | 60.4 ± 1.2 | 10 (1–22) |
| R-GCN | 81.5 ± 6.9 | 72.6 ± 2.1 | 19 (1–37) |
| R-GCN (no sibling) | 81.2 ± 7.6 | 70.9 ± 5.4 | 25 (3–45) |
| GraphSAGE | 67.5 ± 5.0 | 61.8 ± 3.7 | 8 (1–17) |
| GraphSAGE (no sibling) | 74.0 ± 7.9 | 65.3 ± 2.9 | 18 (1–45) |
| GAT | 70.6 ± 6.1 | 61.7 ± 1.2 | 15 (1–43) |
| GAT (no sibling) | 73.2 ± 10.4 | 63.1 ± 1.9 | 16 (2–53) |

## Graph

- `refs+sibling`: relations ['references (fwd)', 'references (rev)', 'sibling (both)']; 1440 directed edges.
- `refs only`: relations ['references (fwd)', 'references (rev)']; 678 directed edges.
- `parent` is in `default_edge_types` but has 0 edges on this corpus (no parent section is a corpus clause), so it contributes no relation.

