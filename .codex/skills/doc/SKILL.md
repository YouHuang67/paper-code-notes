---
name: doc
description: >-
  Write, revise, or review technical and academic documents with rigorous
  logic, disciplined notation, concise main text, and appendix-backed detail.
  Invoke with $doc or use for substantive Markdown, LaTeX, and manuscript work.
---

# Technical Document Craft

Use this skill for substantial technical, mathematical, research, design, or engineering documentation. Apply the user's requested scope and preserve the repository's established terminology and formatting unless an inconsistency must be corrected.

## Host tree placement

Before creating a file or directory, inspect the parent tree (`refs/README.md`, `docs/README.md`, sibling names). Put the document in the subdirectory that already owns that kind of content, and name it like those siblings.

| Typical owner | Name style |
|---------------|------------|
| `refs/research/<topic>/` | Method/topic kebab directories. Date in the document body. |
| `refs/<Project>/` | Pinned upstream clones only. |
| `refs/papers/` | Follow existing children (often `YYYYMMDD_<topic>/`). |
| `docs/research/` | Kebab filenames, no date-prefixed subdirectories, when that is the local style. |
| `docs/plan/` | `YYYYMMDD_<topic>.md` only if that directory already uses dates. |
| `docs/design/` | Stable contract filenames. |

Do not create a stack of `YYYYMMDD_*` folders under `refs/` or `docs/research/` when siblings are already named by method or topic. Prefer updating an existing report over a new dated directory. Dates in plans and paper caches stay confined to those parents.


## Write By Information Role

- Put the central claim, method, decision, or result in the main text. Keep it direct enough that a reader can follow the primary argument without implementation trivia or defensive qualifications.
- Put derivations, exhaustive cases, implementation evidence, code-specific behavior, parameter tables, and other supporting material in appendices when they would interrupt the main line. Cross-reference an appendix from the main text when a reader may need the detail to verify a claim.
- Give every section one job. State what it receives from the preceding section, what it establishes, and what later section uses from it. Do not reintroduce material that an earlier section has already established; use a brief cross-reference instead.
- Use examples, small tables, diagrams, or equations only when they make a relationship more legible than a compact explanation.

## Definitions, Notation, and Reasoning

- Define each mathematical symbol, set, index range, operator, abbreviation, and data object before its first use. Define its type, dimension, domain, or role whenever that matters to interpretation.
- Keep one symbol for one concept. Reuse an established symbol consistently; rename only when two concepts would otherwise be conflated. Never silently change a symbol's meaning, index range, normalization, or scope.
- Before introducing a formula, identify its input objects and the question it answers. After it, state the output or operational consequence needed by the next step.
- Derivations must expose the substitutions or identities that justify a non-obvious result. Move routine algebra and long proofs to an appendix, with a precise reference from the main text.
- Make dependencies explicit. A later construct may rely only on objects already defined, and explanatory prose must preserve the causal order of the algorithm or argument.

## Style and Economy

- Prefer affirmative, concrete prose. Prohibit contrast-by-negation constructions such as "not X but Y" and "rather than X" in every document. State the intended object or behavior directly, then describe relevant distinctions with separate factual sentences.
- 禁止使用“不是……而是……”“并非……而是……”“不是……却……”和“而非……”等否定式对比句。需要区分概念时，分别陈述各自的对象、机制和证据，避免把无关概念强行构造成二选一关系。
- 每份文档初稿完成后，必须检索“不是”“而是”“而非”“并非”“而不”“却”等高风险表达，逐句判断语义；发现对比式否定时改成正向陈述，再进行两轮审阅。
- Remove purpose-only narration, repeated caveats, generic transition sentences, and explanations that add no decision-relevant information.
- Do not sacrifice the essential mechanism for brevity: retain the input, transformation, output, and reason for each central method step.
- Match detail to the reader's needs. Introduce technical depth where it becomes necessary, not prematurely and not after relying on it.

## Editing Workflow

1. Read the surrounding document, the parent directory's siblings, and any local writing rules before editing. Identify the document's argument, section responsibilities, notation inventory, references to appendices or code, and whether a new path would break the host tree.
2. Diagnose structural and notation problems before rewriting prose. Preserve correct existing material and make narrowly scoped edits.
3. When adding a concept, establish it once at its natural introduction point, then use it consistently. Update dependent wording, equations, cross-references, and appendices in the same change.
4. For a substantial revision, perform two separate review passes before delivery.

## Mandatory Two-Pass Review

**Pass 1: logical continuity.** Read every changed sentence in context. Verify that all terms and symbols are defined before use, equations have valid inputs, claims follow from stated premises, sections hand off cleanly, and no later text contradicts or duplicates an earlier definition.

**Pass 2: reader economy.** Read the complete changed flow as a first-time expert reader. Remove redundancy, purposeless framing, negative contrast phrasing, code details that belong in an appendix, and detail that obscures the main argument. Confirm that essential mechanisms, caveats, and appendix references remain discoverable.

Report the two review outcomes briefly, including any residual ambiguity, missing evidence, or validation that could not be completed.
