# Query Authoring and Review Guide

The query-representation experiment found a large input-pipeline effect on the
current 49-query benchmark. All original questions are already single-sentence
questions. Sentence count alone is therefore not a supported dataset-quality
criterion. Use this guide to improve task clarity and coverage while retaining
realistic, difficult questions.

## Preserve the task before improving its wording

Keep the original `query`, `code`, and query ID immutable in the v1 benchmark.
Record candidate wordings separately. The current 49 candidates in
[rewrite_review.csv](../../results/query_diagnostics/rewrite_review.csv) still
require independent semantic review. Retrieval improvement does not establish
equivalence or correct labels.

A reviewer should check these items against the program and specification:

| Check | Acceptance criterion |
| --- | --- |
| Entity references | Every “this”, “it”, and named variable has a clear referent. |
| Requested task | Behavior, value category, sequencing, validity, and evidence retrieval are distinguished. A rewrite does not silently narrow the task. |
| Program fidelity | Code remains complete for the intended reasoning. Types, initialization, overloads, expression boundaries and evaluation context are preserved. |
| Assumptions | Language version and necessary context are explicit. Do not silently assume built-in operators, single-threaded execution or absence of overloads. |
| Neutral wording | Name visible operations where useful; do not reveal the verdict, gold clause names, or evidence text. Ordinary user wording remains represented. |
| Gold support | Each required clause has a supporting paragraph/span and a stated role. Check alternative sufficient evidence sets separately. |
| Evidence budget | Difficulty and required evidence size reflect the reviewed task, not model success. |
| Review outcome | Record reviewer, rationale and accepted/revise/reject status. A scope change creates a new task with independently checked gold. |

Operator names are descriptions of spellings until context resolves their use.
For example, `<<` may occur in shifts or overloaded stream operations, and a comma
can be an operator or a separator. The automatic glossary is an experimental
retrieval feature; it is not an annotation authority.

## Suggested additional metadata

The [review template](query_review_template.json) is an annotation-side record,
not a benchmark replacement or an input accepted by the retrieval CLI. Link it
by `query_id`. Fill its null fields during review.

- `family_id`: shared across paraphrases, contrast programs and source templates.
- `task_intent`: direct lookup, behavior classification, value/value category,
  sequencing, validity, or complete evidence retrieval; allow multiple intents
  when the original task requires them.
- `assumptions`: explicit contextual assumptions, with their source.
- `required_evidence`: clause ID, paragraph/span anchor, and why it is necessary.
- `alternative_evidence_sets`: independently reviewed sufficient alternatives.
- `candidate_variants`: wording, axis changed, semantic review and gold review.
- `split`: assigned at family level, after duplicate/template inspection.

Keep evidence, expected verdicts, difficulty and reviewer notes out of retrieval
inputs. The production baseline uses only `query` and `code`, optionally
appending its deterministic operator glossary.

## Next controlled wording experiment

Use a 2-by-2 design separating verbosity from terminology. For each task, create
concise/plain, concise/technical, expanded/plain and expanded/technical variants.
Each must preserve the same entities, assumptions, requested answer and complete
code. Longer variants may restate the task, but cannot introduce new facts or
generic instructions that change the framing. Record word counts; one-sentence
status is descriptive metadata, not the intervention itself.

Have another reviewer check semantic scope and gold before running retrieval.
Do not choose paraphrases based on retrieval results. Freeze the variant texts,
model revision, BM25 configuration, candidate budget and fusion settings first.
Report paired Recall@10 and complete coverage@10 changes over original query
IDs, stratified by evidence size and task intent. Include all four conditions,
per-task losses and bootstrap intervals; do not count variants as independent
tasks. This study can isolate wording effects more clearly than the existing
explicit-versus-verbose diagnostic.

## Expand coverage with code-dependent contrast families

Prioritize these contrasts after checking that the corpus contains their needed
evidence. Each modified program is a new task and needs its own gold review:

| Family | Controlled change | What to check |
| --- | --- | --- |
| Increment | Prefix versus postfix | Requested value/value category and side effects |
| Operator resolution | Built-in versus user-defined overload | Declarations, overload selection and applicable clauses |
| Logical operators | `&&`/`||` versus `&`/`|` | Operand types, short circuiting and sequencing |
| Comma | Comma expression versus argument separation | Parse context and ordering guarantees |
| Side effects | One versus multiple modifications of an object | Full-expression boundaries and sequencing relations |

Balance direct lookup and multi-clause tasks, ordinary and technical wording,
and both easy and difficult programs. The current labels touch only 11 distinct
clauses despite the 82-clause corpus; broaden reviewed topic coverage rather
than adding only more variants of existing successful queries.

Keep every family in one split. Treat the current 49 queries as exploratory
development data because they have already informed pipeline choices. Freeze a
new family-separated test set before evaluating the chosen pipeline. Retain
failures and ambiguous cases with explicit review status; do not delete them to
improve reported scores or to favor graph retrieval.
