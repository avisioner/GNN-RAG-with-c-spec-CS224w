# Further Query and Dataset Experiments

The goal is to understand the intended task and retrieval robustness, then decide
whether to improve the input pipeline, the dataset, or the retrieval method.
Selecting only questions a particular model finds easy would not answer that
question.

## First Round Implemented Here

The fixed nine-condition paired experiment covers question/code information,
field formatting and order, domain framing, symbolic operator verbalization,
explicit single-question wording and added verbose framing. The original
questions are already one sentence each, so their sentence count has no
variation to correlate with performance.

## Follow-Up Experiments Requiring Reviewed Data

| Axis | Matched comparison | Required control | What it diagnoses |
| --- | --- | --- | --- |
| Concision | One concise sentence vs several fluent sentences | Identical facts, task, code, assumptions and gold; avoid adding hints in the longer version | Sensitivity to verbosity rather than evidence availability |
| Vocabulary | Everyday wording vs specification terminology | Same semantic scope; independently check paraphrases | Lexical familiarity or terminology dependence |
| Coreference | "this expression/the two updates" vs explicit entity/operator names | Resolve references using the supplied code only | Whether unclear entity references or code understanding are the bottleneck |
| Identifier names | Neutral variable/function names vs descriptive names | Rename all references consistently, including in the question | Spurious identifier cues or dependence on naming conventions |
| Code formatting | Compact, formatted and commented versions | Same token-level program; no explanatory answer-bearing comments | Formatting/tokenization sensitivity |
| Code context | Full program vs minimal typed expression | Preserve declarations, overloads, value categories and scope; prefer compiler/AST-assisted reduction | Boilerplate dilution vs essential typing/context information |
| Task phrasing | "explain behavior" vs a precise evidence request | Treat as separate task intents if the requested evidence differs | Whether a benchmark mixes lookup, semantic classification and full evidence retrieval |
| Context completeness | Explicit edition, platform and validity assumptions vs missing context | Gold re-review if semantics or applicable rules change | Genuine dataset ambiguity rather than model sensitivity |
| Distraction | Relevant question plus neutral background vs relevant question alone | Do not accidentally add a second task or change the program | Robustness to realistic user context |
| Cross-family generalization | Familiar vs held-out operator/code families | Split by original example, source test, template and all paraphrases | Leakage and reliance on a small set of clause priors |

Do not assume these are label-preserving transformations. For example, removing
declarations can remove an overload, changing an integer type can change valid
behavior, and changing the C++ edition can change the applicable rules.

## Contrast Sets for Code Dependence

Include paired tasks with similar natural-language questions but a meaningful
code change: prefix/postfix update; built-in/overloaded operator; comma expression
versus function-argument separator; logical versus bitwise operator; independent
objects versus the same object; or different declared types. Review the gold
evidence for each case separately. Their gold sets or expected behavior may
legitimately differ.

A question-only model can score well when wording identifies a clause without
distinguishing the actual programs. Contrast-set accuracy and evidence-set
changes can reveal this failure mode even when aggregate Recall@10 is high.

## Dataset Annotation Fields Worth Adding

- `task_intent`: direct lookup, behavior classification, ordering relation, or
  retrieval of a complete evidence set; annotate independently of model results.
- `assumptions`: standard edition, built-in/overloaded operation, integer-width
  requirements, valid pointers/indices, and relevant overflow preconditions.
- `evidence_spans` or `gold_chunk_ids`, plus a short explanation of why each
  clause is required and whether alternative sufficient evidence sets exist.
- `family_id`, `source_revision`, and `paraphrase_group_id` for leakage-aware
  splitting and reproducibility.
- `question_style`, `code_required`, and `semantic_review_status` for balanced
  reporting and review, rather than for model input.

## How to Decide What to Change

| Observation | First action |
| --- | --- |
| Labels/fences help while facts are unchanged | Improve input serialization in the pipeline; the dataset may be fine |
| Explicit operator wording helps | Determine whether real users are expected to provide that wording; evaluate input-side code verbalization and reviewed paired annotations |
| A rewrite changes the intended task or evidence | Review the item and gold; do not score it as the same-label paraphrase |
| Missing assumptions allow different answers | Add context and re-annotate before measuring retrieval gains |
| Equivalent reviewed paraphrases have unstable outcomes | Add paraphrase families and report mean/worst-case robustness |
| Graph expansion loses across query styles | Investigate propagation/ranking as well as seeds; do not change data solely to favor a graph method |
| Results differ mainly with gold-set size or task intent | Balance the benchmark and stratify evaluation; do not attribute the effect to sentence length |

Keep difficult, realistic inputs. Report the original condition, each planned
variant, per-family averages, and worst-variant coverage. Put every paraphrase
and related code example in the same future split. Use a development set to
choose a representation and a frozen test set for the final comparison.
