# Pulsrs Development Roadmap

This roadmap is subordinate to [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md), which is the maintained architectural and strategic source of truth.

## Phase 0 — Establish a reviewable baseline

**Goal:** turn the current experimental worktree into a known, reproducible foundation.

- **Completed:** add standard checks for formatting, all-target compilation, strict Clippy, and debug/release tests.
- **Completed:** make planning files trackable and add CI coverage for the baseline checks.
- **Completed:** record provisional benchmark measurements and document the current tensor-core contract.
- **Remaining:** classify tracked and untracked changes.
- **Remaining:** decide which current kernel/SIMD/error/API changes belong in the baseline.
- Record supported behavior and currently panicking behavior.
- Preserve existing passing tests while adding regression tests for newly accepted semantics.

See [`PHASE_0_BASELINE.md`](PHASE_0_BASELINE.md), [`PHASE_0_BENCHMARKS.md`](PHASE_0_BENCHMARKS.md), and [`WORKTREE_CLASSIFICATION.md`](WORKTREE_CLASSIFICATION.md) for exact checks, measurements, and remaining exit criteria.

**Exit gate:** a cleanly described baseline can be built and tested from a fresh checkout, with no ambiguity about experimental files.

## Phase 1 — Harden tensor correctness

**Goal:** make the tensor core reliable across layouts and edge cases.

- Define scalar and empty tensor semantics.
- Audit shape/stride/offset arithmetic and bounds checks.
- Test chained reshape, permutation, slicing, broadcasting, and contiguous conversion.
- Test negative steps, singleton dimensions, high rank, duplicate/invalid axes, incompatible shapes, and integer overflow.
- Standardize typed error paths and reduce avoidable panics.

**Exit gate:** adversarial tests cover every public tensor operation and verify view/copy invariants.

## Phase 2 — Prove memory behavior

**Goal:** make “light and gentle” measurable.

- Instrument allocations where practical.
- Measure peak resident memory for construction, views, elementwise operations, reductions, and materialization.
- Document ownership/lifetime behavior of `Rc<Buffer<T>>`.
- Identify hidden copies and define when copies are allowed.
- Decide the future thread-safe ownership boundary before parallel execution.

**Exit gate:** representative workloads have allocation and peak-memory baselines, with explicit regression thresholds.

## Phase 3 — Optimize execution

**Goal:** approach competitive practical speed without compromising semantics.

- Establish contiguous and strided scalar baselines.
- Centralize elementwise dispatch.
- Complete safe auto-vectorizable kernels and prove equivalence to fallback paths.
- Add architecture-specific SIMD only after measurements justify it.
- Evaluate reductions separately because floating-point reordering affects exact results.
- Add parallel execution only after ownership and synchronization policy are explicit.

**Exit gate:** release benchmarks report throughput, latency, allocation cost, and contiguous/non-contiguous behavior; SIMD results have correctness coverage.

## Phase 4 — Define the ATen-like semantic core

**Goal:** provide the stable backend contract for higher layers.

Prioritize:

- tensor metadata and layout;
- construction and dtype inspection;
- indexing, slicing, views, and contiguous conversion;
- elementwise arithmetic and comparisons;
- reductions and random construction;
- explicit errors and documented unsupported semantics.

Do not chase broad API parity until each supported semantic has tests and memory behavior is understood.

**Exit gate:** the core API and semantic test suite are stable enough for another crate/frontend to depend on.

## Phase 5 — Prepare autograd

**Goal:** make forward operations extensible without prematurely coupling the core to a graph engine.

- Define operation records and gradient-relevant metadata.
- Set mutation, aliasing, view, and saved-intermediate rules.
- Identify which operations can safely record backward functions.
- Implement a small differentiable operator set only after forward semantics are stable.

**Exit gate:** autograd can be layered without rewriting tensor storage or invalidating view semantics.

## Phase 6 — Integrate the larger ecosystem

**Goal:** build the project under the tensor core.

- Python frontend and packaging.
- Neural-network modules.
- Serialization and checkpoint loading.
- Optimizers and training execution.
- Runtime scheduling and optional parallel/GPU backends.

Each layer should consume the stable core rather than introduce hidden tensor semantics.

## Prioritized next work

1. Review and classify the current dirty worktree.
2. Write the tensor-core contract and semantic compatibility matrix.
3. Add missing edge-case tests before optimizing further.
4. Add reproducible format/lint/test/benchmark checks.
5. Measure allocations and memory before claiming lightweight behavior.
6. Complete and validate the safe SIMD Phase 0/1 work.

## Explicit non-goals for the immediate milestone

- Full PyTorch API parity.
- Autograd implementation.
- GPU support.
- Neural-network modules and optimizers.
- Architecture-specific unsafe SIMD before scalar baselines and correctness tests.
