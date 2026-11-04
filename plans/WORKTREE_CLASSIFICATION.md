# Phase 0 Worktree Classification

Status: **classified, split, committed, and locally validated**

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

The corresponding local branches now form a dependency-aware sequence:

| Branch | Intended scope | Depends on |
|---|---|---|
| `phase-0-tooling-context` | CI, planning context, `.gitignore`, formatting/tooling cleanup | `master` |
| `vectorizable-setup` | Vectorization marker and minimal kernel setup | `phase-0-tooling-context` |
| `error-refinement` | Typed error families, conversions, and dependent call-site migration | `vectorizable-setup` |
| `tensor-core-integration` | Tensor, iterator, layout, mapping, and operation integration | `error-refinement` |
| `benchmarks-examples` | Benchmark targets and example updates | `tensor-core-integration` |

The branches have been selectively staged, validated, and committed in dependency order. The current worktree is clean on `benchmarks-examples`; remote publication remains intentionally separate.

### Commit A — Vectorizable setup only

Eligible contents:

- internal vectorization marker/sealing mechanism;
- kernel executor scaffolding only if it is behavior-preserving and correctly named;
- minimal module registration;
- focused kernel tests proving length and output invariants.

Do **not** include tensor semantic changes, reductions, public operator-bound changes, examples, or benchmark rewrites in this commit.

The vectorizable foundation was committed as `2d5502e` (`feat: add vectorizable kernel foundation`). It is intentionally a setup commit, not a claim that architecture-specific SIMD is complete. Current follow-up blockers remain:

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

The error refinement was committed as `299e458` (`refactor: establish typed tensor error families`). It includes the dependent tensor call-site migration required for the new error families. Current follow-up concerns remain:

- the flat `PulsrsError` variants were replaced by nested families, which is a deliberate public API break and must be explicitly accepted;
- several names contain stable-looking spelling issues (`Uncompatible`, `OutofBounds`, `Uncompabtible`);
- `IteratorError` and `SlicingError` are empty public enums with no current use;
- current error tests do not establish a complete message/variant contract.

### Commit C — Tensor behavior and optimization integration

Keep the tensor operation, iterator, reshape, slicing, broadcasting, reduction, and mapping changes separate. This is the largest semantic portion of the current batch and should not be hidden inside either baseline setup commit.

### Commit D — Tooling and repository context

Keep CI, plans, formatting-only changes, examples, and benchmark harness changes separate from the core semantic commits where practical.

## Eligibility decision

**Current answer: eligible after splitting, not as one commit.** The worktree has now been split into validated local commits:

- `804bea6` — `chore: establish phase 0 project baseline`
- `2d5502e` — `feat: add vectorizable kernel foundation`
- `299e458` — `refactor: establish typed tensor error families`
- `e32d3ce` — `feat: integrate tensor core operations and views`
- `d873216` — `perf: add benchmark and example baseline`

The branches are clean and ordered, but the vectorization names/API and typed-error public spellings still need later review before being treated as final contracts.

## Commands for refreshing this inventory

```text
git status --short
git diff --name-status
git ls-files --others --exclude-standard
```
