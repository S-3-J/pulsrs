# Pulsrs Project Context

> **Purpose:** Internal long-term context for planning and implementation. This is not public documentation. Keep it current whenever architecture, goals, supported semantics, or project status changes.

## 1. Product vision

Pulsrs is the tensor-computation foundation for a larger PyTorch-like deep-learning ecosystem. It is intended to fill the role that ATen fills inside PyTorch: a reusable, efficient tensor core on which autograd, neural-network modules, model loading, optimizers, and training can later be built.

The project has three non-negotiable goals:

1. **Correctness:** tensor semantics, layouts, errors, and numerical behavior must be dependable.
2. **Speed:** Rust should provide low-overhead execution and a path to SIMD and parallel kernels.
3. **Lightweight resource use:** avoid PyTorch-like unnecessary RAM retention, hidden copies, and allocation-heavy abstractions.

“Comparable speed” means comparable practical throughput for supported workloads, not a promise to beat mature PyTorch kernels immediately. Memory use, allocation count, and peak resident memory are first-class performance metrics alongside latency and throughput.

## 2. Current strategic decisions

- **Near-term priority:** stabilize the ATen-like tensor core before expanding into autograd or model training.
- **Compatibility target:** important PyTorch tensor semantics, not strict API or complete operator parity.
- **Implementation language:** Rust, currently edition 2024.
- **Core design preference:** explicit shape/stride metadata, zero-copy views where possible, and explicit materialization through `contiguous()`.
- **Current ownership model:** `Rc<Buffer<T>>`; this is suitable for local single-threaded sharing but is not yet the final parallel/runtime ownership boundary.
- **Known intentional limits:** broadcasting, autograd, GPU support, and production-level performance are not complete goals of the current baseline.
- **Planning rule:** correctness and a measurable baseline precede unsafe SIMD, broad API expansion, and higher-level integration.

## 3. Repository map

### Core crate

- `src/lib.rs` — crate documentation, public module exports, and the `function_signature!` helper.
- `src/prelude.rs` — convenience exports.
- `src/tensor/` — `Tensor<T>` and tensor operations.
- `src/buffer/` — one-dimensional `Vec<T>` storage wrapper.
- `src/shape/` — dimensions, rank, element count, names, and index validation.
- `src/stride/` — row-major stride construction and offset calculation.
- `src/cursor/` — index and offset traversal.
- `src/iterator/` — C-style, indexed, strided, and custom-priority iteration.
- `src/slicing/` — slice-range parsing and conversion.
- `src/dtype/` — `DType` and the `Element` trait.
- `src/error/` — shape, tensor, memory, mapping, kernel, and bad-practice errors.
- `src/random/` — seeded random construction support.
- `src/kernels/` — current internal kernel executor scaffolding.
- `src/simd/` — current `Vectorizable` marker and planned vectorization boundary.

### Tensor representation

`Tensor<T>` currently contains:

- shared `Rc<Buffer<T>>`;
- logical `Shape`;
- explicit `Stride` values;
- element `offset`;
- `contiguous` metadata;
- `DType` metadata.

This separates storage from interpretation. A view can share the same allocation while changing shape, strides, and offset. Permutations, slices, and broadcasts are intended to remain zero-copy; `contiguous()` materializes a new row-major buffer when needed.

### Current tensor API areas

- Construction: vector, scalar, zeros, ones, seeded random, ascending/descending ranges, named dimensions.
- Metadata: rank, dimensions, strides, number of elements, scalar status, offset, dtype, contiguity.
- Layout/view operations: reshape, view, flatten, permutation, slicing, broadcasting, contiguous conversion.
- Computation: unary maps, function maps, binary elementwise maps, add/subtract/multiply/divide, equality, reductions.
- Reductions: sum, product, min, max, mean, including arbitrary axis lists through custom-priority iteration.
- Access and traversal: indexing, tensor iterators, index iterators, display.

## 4. Validation and project surfaces

- Integration tests exist for buffers, shapes, strides, cursors/iterators, slicing, and tensors.
- Examples cover construction, operations, shapes, slicing, and broadcasting.
- Criterion benchmarks cover reshape, tensor operations, and contiguous/non-contiguous operations.
- `plans/python_fe.md` describes a phased Python frontend.
- `plans/simd_plan.md` describes baseline measurement, safe auto-vectorizable kernels, architecture-specific kernels, reduction acceleration, and adversarial SIMD tests.
- `plans/PHASE_0_BASELINE.md` records the reproducible validation matrix and worktree classification.
- `plans/PHASE_0_BENCHMARKS.md` records provisional optimized-profile measurements; they are not yet regression thresholds.
- `plans/TENSOR_CORE_CONTRACT.md` records the draft current semantic contract and open decisions.
- `plans/WORKTREE_CLASSIFICATION.md` separates baseline infrastructure from the experimental implementation batch.
- Current validation: formatting, all-target compilation, strict Clippy, debug tests, and release tests pass in the current worktree. The full all-target local matrix also passes, including examples and benchmark harnesses.

## 5. Requirement status snapshot

| Requirement | Status | Evidence / interpretation |
|---|---|---|
| Rust tensor core | **Substantially present** | Generic `Tensor<T>`, `Buffer`, `Shape`, `Stride`, iterators, mappings, views, and errors exist. |
| ATen-like metadata/layout foundation | **Partially present** | Shape, stride, offset, contiguity, permutation, reshape, slicing, and broadcasting exist; semantic edge coverage still needs hardening. |
| Zero-copy views | **Partially present** | Shared `Rc<Buffer<T>>` is used by views; chained and offset-aware cases need stronger tests and contract documentation. |
| Correct indexing/slicing | **Partially present** | APIs and tests exist, including negative-step slicing; invalid and complex chained layouts need broader adversarial coverage. |
| Broadcasting | **Partially present** | Shape compatibility and zero-stride views exist; behavior and error semantics need a stable contract. |
| Elementwise execution | **Present, experimental optimization** | Contiguous and non-contiguous paths exist; current kernel scaffolding is safe iterator-based execution, not architecture-specific SIMD. |
| Reductions | **Present** | Sum/product/min/max/mean and custom-axis iteration exist; empty tensors, numerical semantics, and performance need explicit policy/tests. |
| Dtype foundation | **Present for built-ins** | `DType` and `Element` cover common integer, float, and bool types; promotion/conversion semantics are not yet a full policy. |
| Error model | **Partially present** | Typed error families exist, but several APIs still panic and error behavior is not uniformly fallible. |
| Memory efficiency | **Unverified** | Design favors views, but allocation counts, peak memory, and hidden-copy behavior are not systematically measured. |
| Comparable speed | **Unverified** | Benchmarks and kernel scaffolding exist; no accepted baseline or architecture-specific SIMD evidence is recorded. |
| Python frontend | **Planned** | Detailed plan exists; no binding/package implementation is present in the repository. |
| Autograd | **Not implemented** | It is a long-term direction; operation metadata and mutation/aliasing rules are not finalized. |
| Neural-network/training stack | **Not implemented** | No module, optimizer, serialization, or training runtime exists yet. |
| GPU support | **Out of scope for current phase** | Do not treat absence as a current defect. |

## 6. Important current-state caveat

The working tree is substantially dirty. It contains modified tracked files and untracked additions for error types, kernel/SIMD scaffolding, benchmarks, examples, and functional operations. Treat the current state as an active development snapshot, not as a clean released baseline. Before adding major features, classify and review these changes so the project has a reproducible foundation.

## 7. Engineering invariants to preserve

- Shape accurately describes logical dimensions.
- Strides and offsets map logical indices to the correct storage elements.
- Views share storage unless an operation explicitly promises/materializes a copy.
- Non-contiguous layouts remain non-contiguous and use a correct fallback path.
- `contiguous()` produces row-major storage and correct metadata.
- Scalar, empty, singleton, high-rank, and non-contiguous tensors have defined behavior.
- Fallible APIs return typed errors; panics are reserved for explicitly infallible/operator-style APIs.
- Fast paths must be behaviorally equivalent to scalar/strided fallbacks.
- Memory and performance claims must be backed by measurements, not inferred from implementation shape.

## 8. Maintenance protocol

Update this file when:

- a public semantic is added, removed, or changed;
- an operation changes from copy to view or view to copy;
- dtype, promotion, error, mutation, or threading policy changes;
- a roadmap milestone is completed;
- benchmark or memory evidence changes a performance conclusion;
- a planned limitation becomes supported or is intentionally deferred.

When updating, preserve the status table and add a dated or commit-linked note to the change history below.

## 9. Change history

- **2026-11-04:** Initial internal context created from repository inspection and project-owner direction. Tensor foundation is substantial; memory efficiency, comparable speed, Python integration, autograd, and training layers remain incomplete or unverified.
- **2026-11-04:** Phase 0 validation infrastructure added. Formatting, all-target compilation, strict Clippy, debug tests, and release tests pass; worktree classification and benchmark recording remain open.
- **2026-11-04:** Provisional benchmark measurements, a draft tensor-core contract, and a worktree classification were added. Acceptance review remains open.
- **2026-11-04:** Review concluded that the current batch is not eligible as one “vectorizable setup + error refinement” commit; it must be split because it mixes experimental kernels/errors with broad tensor semantics and tooling.
