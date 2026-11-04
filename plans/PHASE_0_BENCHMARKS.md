# Phase 0 Benchmark Baseline

Status: **recorded provisional baseline**

Last verified: **2026-11-04**

## Method

Benchmarks were run in the optimized Cargo profile on the current worktree using Criterion and the Plotters backend because Gnuplot was unavailable.

The broad sweep used:

```text
cargo bench --all-features --bench contiguous_tensor_operations_benchmark -- --noplot --sample-size 10 --warm-up-time 0.1 --measurement-time 0.5
cargo bench --all-features --bench tensor_operations_benchmark -- --noplot --sample-size 10 --warm-up-time 0.1 --measurement-time 0.5
cargo bench --all-features --bench tensor_reshape_benchmark -- --noplot --sample-size 10 --warm-up-time 0.1 --measurement-time 0.5
```

The selected reference cases used Criterion’s default 100 samples:

```text
cargo bench --all-features --bench contiguous_tensor_operations_benchmark -- "Contiguous Tensor Operations/Contiguous Addition Arange BM/1000000" --noplot
cargo bench --all-features --bench tensor_operations_benchmark -- "Permuted Addition BM/250000" --noplot
cargo bench --all-features --bench tensor_reshape_benchmark -- "Reshape on Non-Contiguous" --noplot
```

These are machine-local observations, not cross-machine performance claims. CPU model, OS scheduling, Rust version, compiler flags, and background load must be recorded before comparing runs.

## Selected reference measurements

Criterion reports `[low, median, high]`:

| Workload | Result |
|---|---:|
| 1,000,000-element contiguous `f32` addition | `[260.02 µs, 260.83 µs, 261.70 µs]` |
| 250,000-element permuted addition | `[9.8749 ms, 9.9756 ms, 10.091 ms]` |
| Non-contiguous reshape benchmark | `[3.7160 ms, 3.7401 ms, 3.7646 ms]` |

## Broad sweep representative medians

The shortened sweep is useful for shape and operation coverage, but its 10-sample configuration is exploratory:

| Workload family | Size | Representative median |
|---|---:|---:|
| Contiguous `f32` addition | 100,000 | 61.621 µs |
| Contiguous `f32` addition | 250,000 | 127.12 µs |
| Contiguous `f32` addition | 1,000,000 | 269.51 µs |
| Contiguous `f64` addition | 1,000,000 | 572.05 µs |
| Contiguous + broadcasted addition | 250,000 | 8.8255 ms |
| Permuted addition | 250,000 | 11.490 ms |
| Contiguous reshape | benchmark case | 100.24 ns |
| Non-contiguous reshape | benchmark case | 3.9817 ms |

The broad sweep also covers subtraction, multiplication, and division for contiguous `f32`/`f64` tensors.

## Interpretation

- Contiguous elementwise work is substantially faster than the current permuted/broadcasted workloads, as expected from the layout-aware execution paths.
- The current numbers include allocation and tensor-construction behavior inside the benchmarked expression where the benchmark code does so; they are not pure kernel throughput measurements.
- No SIMD claim is made. The current kernel scaffolding is iterator-based and must be validated at the generated-code level before being called vectorized.
- No memory-efficiency claim is made. These measurements do not report allocation count, peak resident memory, or retained buffer lifetime.

## Baseline limitations

- The selected cases are not yet pinned to a recorded CPU/toolchain identity.
- The benchmark suite does not yet separate allocation cost from steady-state operation cost.
- The broad sweep uses a deliberately shortened sample configuration.
- No comparison against PyTorch or another reference implementation has been performed.
- No regression threshold has been chosen.

## Next benchmark work

1. Record `rustc -Vv`, CPU model, OS, and relevant compiler/profile settings.
2. Add allocation-free or preallocated measurement cases where the API permits.
3. Keep the selected 100-sample cases as smoke/regression benchmarks.
4. Add explicit contiguous, permuted, sliced, and broadcasted cases with consistent element counts.
5. Add memory and allocation measurement before claiming that Pulsrs is lightweight.
