# Query Representation Results

We ran a paired diagnostic on all **49 original questions × 9 input representations × 4 retrieval methods**. No model was trained, no graph setting was tuned, and no canonical benchmark or existing milestone result was changed. The original question-only rankings were reproduced for all 49 questions, and the original graph-expansion coverage results also match the milestone table per query.

The main finding is **sensitivity to how code operations are expressed in text**, rather than evidence that a query should simply be one sentence. All 49 existing natural-language questions are already single-sentence questions.

## Most Informative Comparisons

All rows retain the same underlying programs and provisional gold sets, except that question-only/code-only conditions deliberately remove information. The explicit rewrites remain unreviewed diagnostic candidates.

| Input representation | Dense R@10 | Dense Cov@10 | Hybrid 2:1 R@10 | Hybrid 2:1 Cov@10 |
| --- | ---: | ---: | ---: | ---: |
| Original question only | 74.15% | 57.14% | 69.90% | 51.02% |
| Original question + code | 70.92% | 53.06% | 70.24% | 51.02% |
| Labeled question + fenced code | 63.44% | 51.02% | 73.64% | 53.06% |
| Explicit one-sentence question + unchanged code | 84.35% | 67.35% | 91.50% | 79.59% |
| Same explicit question/code + verbose generic framing | 80.10% | 59.18% | 88.44% | 75.51% |
| Original question/code + automatic operator glossary | 84.86% | 67.35% | 91.16% | 79.59% |
| Code only | 22.62% | 6.12% | 20.92% | 8.16% |

The full nine-condition table, including code-first order and a C++23 domain prefix, is in [RESULTS.md](RESULTS.md).

### Operator verbalization gives the clearest pipeline signal

The automatic glossary reads only code token spellings: for example, `++` becomes “increment”, `+=` becomes “addition assignment”, and `&&` becomes “logical AND”. It does not inspect gold clauses, annotations, expected behavior, or retrieved text. It preserves the original question and full code.

Relative to raw question+code, the glossary changes:

- Dense Recall@10 by **+13.95 percentage points**, with 15 per-query wins, 1 loss and 33 ties; paired bootstrap 95% interval **[+6.80, +22.11]** points.
- Hybrid Recall@10 by **+20.92 points**, with 20 wins, 1 loss and 28 ties; interval **[+12.76, +30.27]** points.
- Hybrid complete coverage@10 from **25/49 to 39/49**: 15 wins, 1 loss and 33 ties. The net gain is 14 covered queries.

These are exploratory intervals without multiple-comparison correction, not held-out estimates. They support testing an input-side code verbalization step before concluding that the original dataset must be rewritten. The current glossary is deliberately simple, and operator spellings can have context-dependent meanings in broader C++ code.

### Explicitness matters, but candidate rewrites are not validated new benchmark items

The authored explicit condition names visible operations and entities instead of relying on phrases such as “updating a twice” or “this expression”. It retains the complete original program and does not state a behavior verdict. Hybrid reaches 91.50% Recall@10 and 39/49 complete coverage@10.

That intervention bundles operator wording, repeated snippets, domain framing and sometimes task clarification. It cannot isolate sentence length or guarantee that an underspecified original task has exactly the same scope. Independent semantic review is still required. In particular, Dense loses gold recall on `specqa-021`, `specqa-011`, and `specqa-036`; clearer-looking wording does not uniformly improve retrieval. Candidate questions and a pending-review checklist are preserved for review, rather than installed as canonical data.

The glossary condition is useful because it produces a similar gain through a deterministic input transformation, without hand-authoring a question using benchmark outcomes.

### Longer framing does not establish that shorter queries are intrinsically better

Adding neutral generic instructions to the same explicit question/code lowers Dense Recall@10 by 4.25 points and Hybrid by 3.06 points. However, the paired intervals include zero: Dense **[-9.18, +0.34]**, Hybrid **[-8.84, +2.04]** points. The added words also change framing, not just sentence count.

The appropriate conclusion is to keep inputs focused and test reviewed length-matched paraphrases. It would be premature to impose “one sentence only” as a dataset quality rule.

### Formatting and information effects depend on the retriever

Labels and code fences lower Dense Recall@10 but improve Hybrid slightly. Reversing the two labeled fields has little aggregate effect on Dense recall; BM25 rankings are exactly invariant because its token multiset is unchanged. A domain prefix alone is not a consistent Dense coverage improvement.

Code-only performance is low, but removing the question removes the requested task: the same program can support different questions and evidence sets. This is an information ablation, not proof of poor dataset quality or equivalent-query fragility.

## Why Existing-Query Correlations Need Controls

Using a transparent regex for operator/type/sequence vocabulary, original question-only Dense Recall@10 is 87.82% on 26 cue-bearing questions and 58.70% on the other 23. **All 15 single-clause questions are in the cue-bearing group.** Part of that difference is task complexity rather than wording.

Within the 25 two-gold queries, cue-bearing questions have 81.25% recall (n=8), compared with 58.82% (n=17). This is suggestive but observational. Within three-or-more-gold queries, the small cue-bearing group does not outperform the other group. Length and vocabulary summaries should therefore be reported within evidence-size and task-intent strata, rather than pooled into a claim that short or technical questions are always easier.

## Does This Explain the Graph Result?

Input representation changes the fixed graph method's relative behavior, but it does not show a broad graph-expansion advantage:

- Original question only: GraphExp has 0 coverage@10 wins and 3 losses against Dense.
- Raw question+code: 2 wins and 1 loss.
- Explicit one-sentence wording: 0 wins and 0 losses in complete coverage@10.
- Automatic glossary: 1 win and 0 losses.

Better representation improves seed retrieval, but the current propagation/ranking method still needs separate analysis. Dataset changes should clarify the intended tasks and improve balance, not be selected solely to make graph methods win.

## Recommended Next Steps

1. **Pilot the deterministic glossary as a pipeline feature**, keeping the canonical question/code and evaluating it on future development data. Do not replace the published reference score with the best diagnostic score as though it were a held-out result.
2. **Add reviewed query variants as separate fields**, with original wording preserved. Check entity references, operator descriptions, task scope and assumptions. The file `rewrite_review.csv` contains all 49 candidates, initially marked pending.
3. **Annotate task intent, assumptions and supporting spans.** Distinguish direct lookup, behavior classification, ordering questions and complete evidence retrieval. Validate paragraph support and alternative sufficient evidence sets.
4. **Build code-dependent contrast sets and broaden clause coverage.** Prefix/postfix, built-in/overloaded, comma expression/argument separator, and logical/bitwise contrasts can reveal whether a model distinguishes programs rather than recognizing question vocabulary. Review each contrast's gold independently.
5. **Keep paired variants in the same data split**, together with related source tests and code templates. Report original, variant-average and worst-variant results on frozen test; do not treat 441 representations as 441 independent tasks.

Further brainstormed axes and decision rules are in [EXPERIMENT_IDEAS.md](../../experiments/query_diagnostics/EXPERIMENT_IDEAS.md). The full report is [RESULTS.md](RESULTS.md), with exact retrieval inputs, full scores, per-query metrics and paired statistics in this directory. Model and graph parameters, source hashes, the 49 reference checks and token-length checks are recorded in `run_config.json`.

The glossary is now integrated into the normal baseline; see the [production results](../baseline/operator_gloss/RESULTS.md). All reports and saved runs are indexed in [Results](../README.md).
