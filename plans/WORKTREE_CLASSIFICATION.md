# Phase 0 Worktree Classification

Status: **classified by category; acceptance/commit decision remains open**

The current worktree contains a large development batch relative to the last commit. This file prevents those changes from being mistaken for a clean, released baseline.

## Baseline infrastructure accepted for retention

- `.gitignore` correction so `plans/` and `.github/` are trackable.
- `.github/workflows/ci.yml` with formatting, compilation, Clippy, debug-test, and release-test gates.
- `plans/` internal context, roadmap, requirements, baseline, benchmark, and contract files.
- Formatting and strict-Clippy cleanup in examples, tests, and benchmarks required for the baseline gates.

## Existing implementation changes requiring review

The following modified tracked areas are part of the active experimental tensor-core batch and require a deliberate acceptance decision:

- `Cargo.toml` and `Cargo.lock`: dependency and target changes.
- `src/buffer`, `src/cursor`, `src/dtype`, `src/error`, `src/iterator`, `src/shape`, `src/slicing`, and `src/stride`: storage, metadata, traversal, and error changes.
- `src/tensor`: constructors, mappings, arithmetic, indexing, layout operations, reductions, display, and functional operations.
- `src/lib.rs` and `src/prelude.rs`: public module/export changes.
- `examples/`: API usage changes.
- `tests/`: expanded/modified behavior coverage.

## New experimental files requiring review

- `src/buffer/buffer_types.rs`
- `src/error/{bad_practices,kernel_error,mapping_err,memory_err,shape_err,tensor_err}.rs`
- `src/kernels/mod.rs`
- `src/simd/mod.rs`
- `src/tensor/functional_operations.rs`
- `benches/*.rs`
- `bechmark_ideas.txt`

## Review questions

1. Which error modules and kernel/SIMD scaffolding belong in the first accepted baseline?
2. Are current public bounds and panic paths intentional, or should they be Phase 1 work?
3. Are the benchmark targets stable enough to keep, and should the misspelled `bechmark_ideas.txt` be renamed?
4. Should the current API expansion be committed as one batch or separated into storage, semantics, and optimization changes?
5. Which tests represent required contract behavior versus exploratory implementation tests?

## Proposed clean-start commit split

The corresponding local branches have been created from `master`:

| Branch | Intended scope | Depends on |
|---|---|---|
| `phase-0-tooling-context` | CI, planning context, `.gitignore`, formatting/tooling cleanup | `master` |
| `vectorizable-setup` | Vectorization marker and minimal kernel setup | `master` |
| `error-refinement` | Typed error families, conversions, and error tests | `master` |
| `tensor-core-integration` | Tensor, iterator, layout, mapping, and operation integration | `master` plus accepted prerequisite commits as needed |
| `benchmarks-examples` | Benchmark targets and example updates | `master` plus the API commits they exercise |

These branches are initially pointers to `master`; creating them does not automatically distribute the current dirty worktree. Changes must be selectively staged or applied on the appropriate branch, then validated and committed there.

### Commit A — Vectorizable setup only

Eligible contents:

- internal vectorization marker/sealing mechanism;
- kernel executor scaffolding only if it is behavior-preserving and correctly named;
- minimal module registration;
- focused kernel tests proving length and output invariants.

Do **not** include tensor semantic changes, reductions, public operator-bound changes, examples, or benchmark rewrites in this commit.

Current blockers before accepting the vectorization portion:

- `Vectorizable` is declared public while its module and sealing trait are private; this creates a confusing public-boundary/API situation.
- The `BinarySimdKernel`, `UnarySimdKernel`, and `FunctionSimdKernel` names imply SIMD, but their bodies are ordinary iterator loops and no vector instructions are established.
- `try_new()` cannot fail and the kernel structs contain unused `in_val`/`out_val` state.
- `BinarySimdKernel::operate()` validates input lengths but not output length and can silently truncate writes through `zip`.
- There are no focused tests for kernel length mismatch or output completeness.

### Commit B — Baseline error refinement

Eligible contents:

- typed error-family definitions;
- `Display`/`Error` implementations and `From` conversions;
- mechanical call-site migration;
- focused tests for every public error variant and message.

Current blockers before accepting the error portion:

- the flat `PulsrsError` variants were replaced by nested families, which is a deliberate public API break and must be explicitly accepted;
- several names contain stable-looking spelling issues (`Uncompatible`, `OutofBounds`, `Uncompabtible`);
- `IteratorError` and `SlicingError` are empty public enums with no current use;
- current error tests do not establish a complete message/variant contract.

### Commit C — Tensor behavior and optimization integration

Keep the tensor operation, iterator, reshape, slicing, broadcasting, reduction, and mapping changes separate. This is the largest semantic portion of the current batch and should not be hidden inside either baseline setup commit.

### Commit D — Tooling and repository context

Keep CI, plans, formatting-only changes, examples, and benchmark harness changes separate from the core semantic commits where practical.

## Eligibility decision

**Current answer: not eligible as one commit.** The current worktree is eligible for a clean start only after the batch is split and the proposed branch commits are narrowed to behaviorally justified changes. The existing compile, test, formatting, and strict-Clippy gates are necessary but do not resolve the scope and API-boundary issues above.

## Commands for refreshing this inventory

```text
git status --short
git diff --name-status
git ls-files --others --exclude-standard
```
