---
description: Summary of the N4950 corpus and the 200-query SpecQA benchmark, including the popularity prior, and the process used to add the library rows.
globs: ["**/*.jsonl"]
alwaysApply: false
---

# SpecQA corpus and benchmark

Edition N4950. The corpus is `data/processed/chunks.jsonl`. The benchmark is `data/benchmark/spec_benchmark_v2.jsonl`.

The corpus file is pretty-printed JSON objects concatenated with whitespace. Decode it with `json.JSONDecoder.raw_decode`. The benchmark file is one JSON object per line.

## Corpus

12,581 chunks, 2,387 clauses, built by `scripts/parse_core_spec.py` from the N4950 HTML. Core language (clauses 1–15) is 2,195 chunks and 460 clauses. The standard library (clauses 16–33) is 10,386 chunks and 1,927 clauses. 473 heading-only containers have no chunk. Annexes are not parsed.

A chunk is one clause, or one numbered paragraph when the clause has four or more paragraphs. `chunk_id` is `clause:<clause_id>` or `clause:<clause_id>#p<paragraph>`. `cross_references` holds `{target_clause_id, anchor_text, edge_type}`. `edge_type` is always `references`. There are 13,154 of those links. The R-GCN walks `references` edges, not `parent`.

| `chunk_type` | Count | What it is |
| --- | ---: | --- |
| `clause_subsection` | 10,745 | Numbered paragraph |
| `clause` | 1,255 | Whole clause, including synopses and glossary entries |
| `note` | 581 | Paragraph that is only a note |

`section_id` is the chapter. There are 33 chapters, from `intro.scope` through `thread`. The large library chapters are `containers` (1,618 chunks), `utilities` (1,359), `input.output` (1,118), and `thread` (944).

## Benchmark

200 queries, `specqa-001` through `specqa-200`. The retrieval label is `gold_clause_ids`. `gold_evidence` is one chunk per gold clause, in the same order, and that chunk exists in the corpus.

| Slice | Queries | Distinct gold clauses |
| --- | ---: | ---: |
| `specqa-001`..`specqa-049` | 49 | 11, all in `expr` or `intro.execution` |
| `specqa-050`..`specqa-119` | 70 | 164, disjoint from the 49 |
| `specqa-120`..`specqa-200` | 81 | 152 library clauses, disjoint from the first 119 |
| All 200 | 200 | 327 |

Those 327 clauses are mentioned 427 times. 298 appear once, 18 appear twice, and 11 appear more than twice. The repeats above two are inside the original 49. Library rows were capped at two uses of any clause.

| | Count |
| --- | ---: |
| `single_clause` / `two_clause` / `multi_hop` | 23 / 120 / 57 |
| `well_defined` / `ill_formed` / `undefined_behavior` / `indeterminately_sequenced` | 133 / 57 / 7 / 3 |
| `compile_result` pass / fail | 143 / 57 |
| Source project `llvm-project` / `gcc` | 181 / 19 |

The library slice is 73 `two_clause` and 8 `multi_hop`. It has no `single_clause` row. Properties there are 64 `well_defined`, 16 `ill_formed`, and 1 `undefined_behavior`.

Gold-clause mentions by chapter, highest first: `expr` 99, `basic` 29, `iterators` 26, `utilities` 19, `containers` 19, `class` 17, `over` 16, `temp` 16, `dcl.dcl` 16, `thread` 16, `lex` 15, `stmt.stmt` 14, `cpp` 14, `concepts` 11, `input.output` 11, `library` 10, `except` 8, `time` 8, `ranges` 8, `module` 7, `mem` 7, `strings` 7, `algorithms` 5, `numerics` 5, `localization` 5, `re` 5, `intro` 4, `meta` 4, `intro.defs` 3, `support` 3. `intro.scope`, `intro.refs`, and `diagnostics` have no gold clause.

A row has `query_id`, `query`, `code`, `concept`, `difficulty`, `expected_property`, `spec_edition` (`N4950`), `gold_clause_ids`, `gold_evidence`, `source`, and `validation`. `difficulty` matches the gold count for `specqa-050` onward: 1, 2, or 3 or more. Eight of the original 49 are labeled `multi_hop` with two gold clauses. `ill_formed` fails `clang++ -std=c++23 -fsyntax-only -pedantic-errors -Werror=macro-redefined`. Every other property passes that command. The compiler string on the library rows is `Apple clang version 17.0.0 (clang-1700.6.3.2)`.

## Popularity

The clause-popularity prior is not a model. Queries are split into 5 folds, stratified by `difficulty` with seed 0. For a test query, every clause is ranked by how often it is gold in the other four folds. Ties break by clause id. The score is complete coverage at 10: the query counts only when every one of its gold clauses sits in that top 10.

On the original 49 queries this prior reaches **91.8%** coverage at 10, because those queries share 11 gold clauses and `expr.post.incr` alone is gold in 21 of them. On the full 200-query benchmark the same prior is **20%**. The 70 core queries added in `specqa-050`..`specqa-119` and the 81 library queries are each at 0% on their own: a gold set is no longer a subset of the ten most repeated clauses. Spreading gold across 327 clauses is what makes the shortcut weak.

## How the library rows were added

`specqa-001`..`specqa-119` stayed frozen. The 81 library rows were added in three stages: candidate chains, a prompt file, then authored benchmark rows. `scripts/benchmark_generation/build_and_verify_benchmark_v2.py --write` was not used. That path rebuilds the file from the core `cand-*` table and would replace the frozen rows.

### 1. Candidate subgraphs

`scripts/benchmark_generation/sample_candidate_subgraphs.py --library` reads `chunks.jsonl` and writes `data/benchmark/candidate_subgraphs_library.jsonl`. It does not touch `candidate_subgraphs_v2.jsonl`.

A candidate is a path of one or two `references` edges inside the 18 library chapters. A 2-edge path is preferred and is labeled `multi_hop`. One edge is `two_clause`. The path stops at two edges because a 2-layer R-GCN cannot see a third hop.

An edge is kept only when the citing sentence, after notes and examples are removed, actually imposes a requirement: a shall, an ill-formed, a models or meets, an only-if or unless, or a labeled Mandates, Preconditions, Constraints, Throws, Expects, or Ensures. The sentence must name the target as `[clause.id]` or as a concept token that contains `_` or `-`. A parenthetical definition, a glossary cite, and a hop into `intro.object`, `basic.types.general`, `basic.fundamental`, `expr.const`, or `conv.rank` are dropped. Synopsis clauses are not primaries. The second sentence of a 2-edge path has to share an identifier or a long content word with the first, so the two hops are about the same construct.

Targets that a syntax-only compile cannot decide were blocked: `res.on.data.races`, `concepts.equality`, `res.on.exception.handling`, and `fs.err.report`. Edges rejected in review are stored in `REJECTED_LIBRARY_EDGES` and are not proposed again.

The first rounds took one new chain per chapter (`--per-chapter 1`), appending after the locked rows so a primary clause and an edge are not reused. Those 28 chains (`lib-001`..`lib-028`) were reviewed. Three evidence chunks were corrected (`concept.booleantestable#p1`, `locale.facet#p1`, `swappable.requirements#p2`), one chain was shortened by dropping `concepts.equality`, and the rejected edges above were removed. Ids were then renumbered so they stay sequential. To reach 200 benchmark rows, later rounds used `--per-chapter 6` and one more `--per-chapter 1` pass. The file now holds `lib-001`..`lib-093`. A chapter may be the primary of at most 12 chains, and a clause id may appear in at most two chains once chains already copied into the benchmark are counted once.

Check the file before authoring:

```text
python3.13 scripts/benchmark_generation/build_and_verify_benchmark_v2.py --library --verify
```

That command checks the path, the citing sentence, the chunk ids, the clause cap, and the chapter cap. It prints `failures: []` when the manifest is usable. It does not write queries.

### 2. Prompt file

```text
python3.13 scripts/benchmark_generation/build_and_verify_benchmark_v2.py --library --export-prompts data/benchmark/library_stage2_prompts.jsonl
```

`data/benchmark/library_stage2_prompts.jsonl` is one JSON object per candidate, not a benchmark row. The export of the reviewed 28 chains is what is stored: `lib-001` through `lib-028`. Each object has `candidate_id`, `difficulty`, `gold_clause_ids`, `chapters`, `subgraph_pattern`, `citation_edges`, and `prompt`.

The prompt is the instruction for the next stage. It quotes the citing sentence of each hop and asks for `query`, `code`, `concept`, `expected_property`, and `source`. It says to look in `libcxx/test/std/` or `libstdc++-v3/testsuite/`, to skip `libcxx/test/libcxx/`, to keep a later clause even when no single test file covers the whole path, and to avoid clause ids and the words "this code" in the question. `Mandates` and `Constraints` are `ill_formed`. `Preconditions` are `undefined_behavior` unless the sentence says the program is ill-formed.

### 3. Append to the benchmark

Each prompt was filled by hand and compiled before the row was appended.

The program is a short complete snippet that includes the header the clause names. `ill_formed` must fail the clang++ command above, and the diagnostic must be the rule in the citing sentence. A missing header, a missing member, or a typo is a bad snippet, so that candidate stays in the candidate file and is not appended. Every other property must pass. A precondition that the compiler does not diagnose is `undefined_behavior` with `compile_result` `pass` when the snippet violates it, and `well_defined` when the snippet satisfies it. Runtime checks such as `format` of `%q` or `regex_match` with `match_not_bol` are syntax-only: the stored property is `well_defined` when the program is well-formed.

`query` names the construct and still makes sense with the code hidden. `gold_clause_ids` and `gold_evidence` are copied from the candidate, in order. `source` points at a real `libcxx/test/std` file. The commit is the local llvm-project tree that contains that file, `d0a9d2f12c154a5d32425dbc6c91ee799dc36737`, which is newer than the commit on `specqa-001`..`specqa-119`. `line_start` is a line in that test that mentions the API. `scripts/cpp23_benchmark_seed.json` was only a directory index for finding those tests.

Accepted objects were appended as single JSON lines, starting at `specqa-120`. Of the 93 candidates, 81 became rows. The 12 that did not are `lib-002`, `lib-009`, `lib-017`, `lib-030`, `lib-033`, `lib-055`, `lib-056`, `lib-061`, `lib-062`, `lib-072`, `lib-079`, and `lib-091`. This Apple clang has no `<stacktrace>`, `<generator>`, or `<spanstream>`, and this libc++ has no `views::zip_transform`, `views::join_with`, `views::slide`, `clock_cast`, `is_clock`, or `basic_const_iterator`. `common_iterator<int*, default_sentinel_t>` does not satisfy `sentinel_for` here. `contents` → `headers` and `support.signal` → `stmt.dcl` do not produce a diagnostic of that rule.
