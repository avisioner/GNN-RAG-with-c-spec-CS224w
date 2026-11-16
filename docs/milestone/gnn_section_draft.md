# GNN reranker (draft for the milestone report)

We replaced the fixed graph-expansion heuristic with a learned reranker. Each query gets its own copy of the 82-clause graph. Each clause has seven query-dependent features: z-scored dense max-chunk, dense mean-chunk and BM25 scores, reciprocal ranks under dense and BM25, a dense-top-5 flag, and log degree. There are no clause-identity features. The main model is a two-layer R-GCN (Schlichtkrull et al., 2018) with three relations: references, reversed references, and sibling. It can be read as a learned, relation-aware version of Dense + GraphExp. It is trained with a multi-positive listwise softmax loss over all 82 clauses. The controls share the same features and training setup: an MLP with no message passing, GraphSAGE and GAT (which ignore edge types), and an R-GCN without sibling edges.

We trained on the 49 real benchmark queries only, with no generated data, using 5-fold cross-validation stratified by difficulty. For each test fold, three folds are used for training and one for early stopping, so every query is scored by a model that never saw it. Hyperparameters were fixed in advance.

| Method | Cov@10 | R@5 | Multi-hop Cov@10 |
|---|---:|---:|---:|
| Dense | 57.1 | 60.9 | 11.8 |
| Hybrid (RRF 2:1) | 51.0 | 65.8 | 11.8 |
| Dense + GraphExp | 51.0 | 61.7 | 5.9 |
| MLP (no graph) | 55.1 ± 1.7 | 60.4 ± 1.2 | 15.7 ± 5.5 |
| GraphSAGE | 54.4 ± 1.9 | 61.8 ± 3.7 | 13.7 ± 2.8 |
| GraphSAGE without siblings | 61.9 ± 2.5 | 65.3 ± 2.9 | 29.4 ± 9.6 |
| GAT | 52.4 ± 3.8 | 61.7 ± 1.2 | 15.7 ± 11.1 |
| GAT without siblings | 56.5 ± 2.5 | 63.1 ± 1.9 | 19.6 ± 5.5 |
| R-GCN | 73.5 ± 1.7 | 72.6 ± 2.1 | 58.8 ± 0.0 |
| R-GCN without siblings | 74.1 ± 7.9 | 70.9 ± 5.4 | 49.0 ± 14.7 |
| Clause-popularity prior (no model) | 91.8 | 64.1 | 88.2 |

*49 queries; model rows are mean ± std over 3 seeds.*

The R-GCN beats Dense on coverage@10 by 16 points (95% CI [+4, +29]), mostly on multi-hop queries (+47, CI [+26, +69]). It also beats the MLP by 18 points (CI [+10, +28]). GraphSAGE and GAT perform at the MLP's level, so the gain appears only when edge types get separate weights.

We ran every architecture with and without sibling edges. Siblings make up more than half of the directed edges (762 of 1,440), and they connect clauses that share a parent section rather than clauses that cite each other. Untyped models get worse with them: GraphSAGE loses 7.5 points of coverage@10 (CI [−12.2, −3.4]) and 15.7 on multi-hop, and GAT loses 4.1 (CI [−9.5, +0.7]). The R-GCN is unaffected (−0.7, CI [−7.5, +5.4]). When siblings are averaged together with references, they dilute the reference signal. The R-GCN gives siblings their own weight matrix, so it can down-weight them. Siblings don't help the R-GCN either, but keeping them costs it nothing. Even without siblings, the R-GCN (74.1) beats GraphSAGE (61.9), so keeping forward and reverse references as separate relations also matters.

These gains should not be read as the model learning the spec's structure, because of a problem with the benchmark. Across all 49 queries only 11 distinct clauses are ever gold, and `expr.post.incr` is gold in 21. A baseline that ignores the question and ranks clauses by how often they are gold in the other folds reaches 91.8% coverage@10, beating the R-GCN by 18 points (CI [−33, −4]). Popular clauses fill 70% of the R-GCN's top 10, against 41% for Dense and 49% for the MLP. Our reading is that on a fixed graph a node's typed neighborhood is enough to identify it, so the R-GCN can learn which clauses are popular even without identity features, while the MLP cannot. With 49 queries, the benchmark rewards memorizing popular clauses more than retrieval, so it cannot yet tell us whether the graph helps.

**Next steps.**
- Expand the benchmark so gold labels cover many more distinct clauses, which makes a popularity prior a weak baseline.
- Freeze a dev/test split, with any failure-driven edges curated on dev only.
- Add definition-based `uses_term` edges. For example, `expr.post.incr` says "sequenced before" but never links `intro.execution`.
- Move to the 460-clause corpus.
- Run a rewired-graph control to test directly whether the R-GCN's gain comes from structure or from node identity.

*This analysis and code were produced with the help of Claude (Anthropic), directed and reviewed by Aaron.*
