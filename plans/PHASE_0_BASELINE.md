# Phase 0 Baseline

Status: **baseline checks pass; grouped commit sequence is complete locally**

Last verified: **2026-11-04**

## Purpose

Phase 0 establishes a reproducible engineering baseline before tensor correctness hardening, memory instrumentation, or performance optimization. It does not claim that the current experimental API is stable or production-ready.

## Baseline commands

Run from the repository root:

```text
cargo fmt --all -- --check
cargo check --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test --release --all-features
```

For the complete local target matrix, including benchmark harness compilation/execution:

```text
cargo test --all-targets --all-features
```

The repository CI workflow mirrors the first five checks:

```text
.github/workflows/ci.yml
```

## Verified results

| Check | Result |
|---|---|
| `cargo fmt --all -- --check` | Pass |
| `cargo check --all-targets --all-features` | Pass |
| `cargo clippy --all-targets --all-features -- -D warnings` | Pass |
| `cargo test --all-features` | Pass: 152 tests across integration suites and doctests |
| `cargo test --release --all-features` | Pass: 152 tests across integration suites and doctests |
| `cargo test --all-targets --all-features` | Pass during baseline verification, including examples and benchmark harnesses |

Benchmark output currently falls back from Gnuplot to Criterion’s Plotters backend when Gnuplot is unavailable. This is an environment note, not a test failure.

## Worktree classification

The repository is not a clean baseline commit yet. At the time of this record:

- existing tracked files contain broad uncommitted API, implementation, test, example, and formatting changes;
- untracked files add benchmark targets, error modules, kernel/SIMD scaffolding, buffer types, and functional operations;
- `Cargo.toml` and `Cargo.lock` contain uncommitted dependency/target changes;
- `plans/` and CI are now intentionally trackable repository files.

Until these changes are reviewed and committed or deliberately excluded, “baseline” means the current worktree snapshot plus the checks above, not the last commit.

## Current supported surface

The current implementation provides:

- generic tensor construction and primitive dtypes;
- shape, stride, offset, and contiguity metadata;
- indexing and tensor/index iteration;
- reshape/view/flatten, permutation, slicing, broadcasting, and contiguous materialization;
- unary, binary, functional, arithmetic, comparison, and reduction operations;
- typed error families alongside some panic-based operator/indexing paths;
- internal vectorization markers and iterator-based kernel executors.

## Explicitly not baseline guarantees

- no stable PyTorch/ATen compatibility contract;
- no measured RAM or allocation budget;
- no demonstrated architecture-specific SIMD;
- no thread-safe or parallel runtime ownership model;
- no autograd, Python frontend, neural-network modules, serialization, optimizers, or training runtime;
- no promise that all invalid inputs return errors instead of panicking.

## Phase 0 exit criteria

- [x] Formatting is reproducible.
- [x] All targets compile.
- [x] Strict Clippy passes.
- [x] Debug and release tests pass.
- [x] CI runs the baseline checks.
- [x] Planning files are trackable and indexed.
- [x] Provisional benchmark measurements are recorded.
- [x] Current behavior and unresolved tensor contract decisions are documented.
- [x] Current worktree changes are classified into accepted baseline, follow-up, or discarded work.
- [ ] Tensor-core semantic contract is reviewed and accepted.
- [ ] Baseline benchmark workloads and recorded numbers are accepted as regression references.

## Commit eligibility assessment

The current batch was **not eligible as a single clean-start commit** described as “basic vectorizable setup + baseline error refinement.” It also contained broad tensor behavior changes, public API changes, package metadata changes, examples, tests, benchmarks, CI, and planning files. It has now been split into the ordered local commits recorded in `WORKTREE_CLASSIFICATION.md`.

The recommended clean start is to split it into:

1. vectorizable setup;
2. baseline error refinement;
3. tensor behavior/optimization integration;
4. tooling, benchmarks, and planning context.

See [`WORKTREE_CLASSIFICATION.md`](WORKTREE_CLASSIFICATION.md) for the detailed blockers and proposed boundaries.

The local branch structure for this split is now:

```text
master
└── phase-0-tooling-context
    └── vectorizable-setup
        └── error-refinement
            └── tensor-core-integration
                └── benchmarks-examples
```

The grouped branches have been selectively staged, committed, and validated locally. The current worktree is clean on `benchmarks-examples`.

## Next action

Push the ordered branches only after reviewing the public API breaks and opening the corresponding issues/PRs. Do not interpret the passing checks as proof that tensor semantics, memory use, or performance are already correct.
