# Draft: Specification graph and graph-expansion baseline (Aaron)

*Draft for the milestone report; numbers from `results/milestone/graph_expansion_results.md`.*

## Graph construction

We build a clause graph whose 82 nodes are the corpus clauses (454 chunks from the Expressions, Overloading and Basics chapters of N4950). Edges come from `chunks.jsonl`. A **references** edge joins clause A to clause B when any chunk of A hyperlinks to B. We strip paragraph and term anchors such as `#nt:conditional-expression` from the targets and deduplicate across chunks, which gives 309 undirected edges. We planned **parent** edges from each chunk's `parent_clause` field, but none of the 16 parent values is itself a corpus clause: parents are section headers such as `expr.compound`, whose text lives in `.general` subclauses. We therefore add **sibling** edges between clauses that share a parent, giving 381 more edges. Self-loops are removed, and references that leave the corpus are dropped: 491 of 830 distinct (clause, target) pairs, mostly into the Declarations and Templates chapters. The resulting graph is connected, with median degree 15 and maximum 42 (`expr.const`). It is small-world: on average a clause's 1-hop neighborhood covers 20% of the corpus and its 2-hop neighborhood covers 71%.

Validation (`results/milestone/data_validation.md`) found three issues worth reporting. First, the shared `spec_graph.json` lacks 97 of the 368 clause-level reference edges because its builder did not strip `#` anchors. Its edge set equals exactly the un-anchored subset of the chunk references. Second, 8 of the 17 `multi_hop` queries have only two gold clauses. Third, the 49 queries use only 11 distinct gold clauses, and `expr.post.incr` is gold for 21 of them.

## Method: Dense + GraphExp

Given a query's dense clause scores $d$, we take a clause's score to be the maximum over its chunks; this reproduces Erica's Dense ranking exactly. We min-max normalize to $\tilde d$ and keep the top-5 clauses as a seed vector $s$. We then score

$$\text{score} = \tilde d + \lambda \sum_{t=1}^{h} \hat A^{t} s, \qquad \hat A = D^{-1/2}(A+I)D^{-1/2},$$

with $A$ the undirected adjacency over references, parent and sibling edges. This is a one-step, GCN-normalized relevance diffusion. It raises clauses linked to confident hits, and the degree normalization damps hubs. The setting $\lambda=0.5$, $h=1$, seed\_k $=5$ was fixed before we looked at results. One hop is the default because two hops reach most of the graph.

## Results

| Method | Cov@5 | Cov@10 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| BM25 | 36.7 | 42.9 | 53.1 | 57.8 |
| Dense (BGE-small) | 42.9 | **57.1** | 60.9 | **74.1** |
| Hybrid RRF 2:1 | **49.0** | 51.0 | **65.8** | 69.9 |
| Dense + GraphExp | 46.9 | 51.0 | 61.7 | 69.4 |

*(49 queries, question-only input, complete coverage = all gold clauses retrieved. 95% bootstrap CI for Dense coverage@10 is [42.9, 71.4].)*

Graph expansion did **not** improve over Dense at our headline budget. Against Dense it wins 2 and loses 0 queries on coverage@5, but wins 0 and loses 3 on coverage@10 (paired difference −6.1 points, 95% CI [−14.3, 0.0]). On multi-hop queries it changes the outcome of only one query, a loss. All of these differences are one to three queries, within noise for a 49-query benchmark. Inspecting the changed queries shows the mechanism. Expansion lifts gold clauses sitting just below the cutoff (dense ranks 6–7) into the top 5. It also pushes gold clauses at ranks 8–10 out of the top 10, in favor of high-degree neighbors of the seeds such as `expr.const` (degree 42).

A reachability diagnostic explains why the heuristic is weak. Of the gold clauses Dense misses in its top 5, 91% are 1-hop neighbors of those 5 seeds. However, 56% of *all* non-seed clauses are neighbors too, so the graph says the answer is "nearby" but is only modestly selective: about 1.6× chance with all edges and 2.1× with references alone. Sibling edges are close to chance (36% vs. 31%). Exploratory variants agree. Reference-only expansion has slightly higher recall than the default; using `spec_graph.json`'s reduced reference set comes closest to Dense (coverage@10 55.1); seeding from the hybrid instead of Dense gives 53.1. No λ/hops setting in a small grid beat Dense on coverage@10. We report the grid but choose nothing from it, because doing so would tune on the test queries.

The graph carries the right signal, since missed gold clauses are almost always adjacent to the top hits. Uniform diffusion cannot tell the one relevant neighbor from the many irrelevant ones. That is the gap a query-aware learned reranker is meant to close.

## Plan for weeks 4–5

- **GNN reranker.** An R-GCN over the typed edges (references, sibling, and later term-definition edges). The query enters as per-clause input features: dense score, BM25 score, seed indicator and rank. The model outputs a relevance score per clause.
- **Training data.** Synthetic queries generated from the spec text, with gold sets from the clauses that generated them. We never train on the benchmark. Because the benchmark's gold concentrates on 11 clauses, we will check that the model is not just learning a clause prior.
- **Baselines.** The same features fed to a no-graph MLP, and graph expansion with λ/hops tuned on the dev split, so that any gain can be attributed to message passing.
- **Evaluation.** A frozen dev/test split of the benchmark for model selection, pending Monami's agreement, with coverage@10 as the headline and bootstrap CIs throughout.
