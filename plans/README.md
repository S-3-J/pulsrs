# Pulsrs Internal Planning Index

These files are internal engineering context, not public project documentation.

| File | Purpose |
|---|---|
| [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) | Long-term maintained architecture, strategy, invariants, and current-state context. |
| [`REQUIREMENTS_STATUS.md`](REQUIREMENTS_STATUS.md) | Requirement-by-requirement assessment of what the repository currently fulfills. |
| [`ROADMAP.md`](ROADMAP.md) | Phased implementation roadmap and milestone gates. |
| [`PHASE_0_BASELINE.md`](PHASE_0_BASELINE.md) | Reproducible checks, current baseline results, worktree classification, and exit criteria. |
| [`PHASE_0_BENCHMARKS.md`](PHASE_0_BENCHMARKS.md) | Provisional release benchmark measurements and methodology. |
| [`TENSOR_CORE_CONTRACT.md`](TENSOR_CORE_CONTRACT.md) | Draft current tensor semantics and unresolved contract decisions. |
| [`WORKTREE_CLASSIFICATION.md`](WORKTREE_CLASSIFICATION.md) | Classification of the broad uncommitted development batch. |
| [`python_fe.md`](python_fe.md) | Detailed Python frontend and packaging plan. |
| [`simd_plan.md`](simd_plan.md) | Detailed SIMD, kernel, reduction, benchmark, and validation plan. |

## Maintenance rule

Update `PROJECT_CONTEXT.md` whenever an architectural or semantic decision changes. Update `REQUIREMENTS_STATUS.md` when a requirement moves between absent, partial, present, or verified. Update `ROADMAP.md` when milestones or priorities change. Keep detailed subsystem plans synchronized with those three files.

## Branch workflow

The current grouped branches are documented in [`WORKTREE_CLASSIFICATION.md`](WORKTREE_CLASSIFICATION.md). Create an issue for one branch/group, selectively stage only that scope, validate it independently, commit, push, and merge it before starting the next dependent branch.
