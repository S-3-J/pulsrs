### SIMD Plan for Pulsrs

Context / Problem Statement
All compute flows through mappings.rs over the TensorIterator. Even for contiguous tensors, iteration goes through an enum-dispatched next() → get_from_offset() → Result + bounds check + Rc deref per element. LLVM cannot auto-vectorize this, so contiguous f32 elementwise add runs fully scalar today. The good news: the contiguous: bool metadata and Buffer::as_slice() already exist, so a fast path can be added with one branch and zero public API breakage.

***

Constraint summary: stable Rust only (no std::simd), no new dependencies, unsafe quarantined in tiny kernels, non-contiguous tensors keep the existing iterator path.
- [ ] Phase 0 — Baseline (evidence before optimization)
    - [ ] Add a dedicated criterion bench to benches/tensor_operations_benchmark.rs: large contiguous f32 and i64 tensors (~1M elements) for add/sub/mul/div + scalar-ops. Existing benches cover permuted/broadcast; we need a clean contiguous baseline.
    - [ ] Record numbers before any change. Also keep a manual disassembly spot-check recipe in mind (cargo build --release + objdump -d / otool -tv; look for fadd.4s on this aarch64 machine, vaddps/vaddpd on x86_64) to prove vectorization rather than assume it. No tooling dependency added.
- [ ] Phase 1 — Auto-vectorization-ready slice kernels (no std::arch yet)
    - [ ] New internal module src/simd/ (pub(crate), #[doc(hidden)], registered in lib.rs):
    - src/simd/kernels.rs — a private trait, e.g.:
```rust
pub(crate) trait ElementwiseKernel: Sized + Copy {
    fn add_slices(a: &[Self], b: &[Self], out: &mut [Self]) -> bool { false }
    fn sub_slices(a: &[Self], b: &[Self], out: &mut [Self]) -> bool { false }
    fn mul_slices(a: &[Self], b: &[Self], out: &mut [Self]) -> bool { false }
    fn div_slices(a: &[Self], b: &[Self], out: &mut [Self]) -> bool { false }
}
```
   - false = "not supported, use scalar fallback." Implemented via macro:
        - f32, f64: all four ops.
        - i8..i64, u8..u64: add/sub/mul only; div returns false (no SIMD integer divide on any mainstream ISA).
        - i128, u128, bool: all false.
        Kernel bodies use the form LLVM vectorizes most reliably (zipped equal-length slices → no bounds checks, no Result, no enum dispatch):
        debug_assert_eq!(a.len(), b.len());
        ```rust 
        for ((x, y), o) in a.iter().zip(b.iter()).zip(out.iter_mut()) { *o = *x + *y; }
        ```
- [ ] Tensor-side helper (in metadata.rs or contiguous.rs):
    - pub(crate) fn as_contiguous_slice(&self) -> Option<&[T]> — returns Some(&buffer.as_slice()[offset .. offset + numel]) when self.contiguous and the range is in bounds, None otherwise. Guarded so it's correct even if a future view type marks itself contiguous with a nonzero offset.
- [ ] Rewire dispatch points (minimal intrusion):
    - [ ] try_add/try_sub/try_mul/try_div in binary_operations.rs: when shapes are equal, try as_contiguous_slice() on both operands; if both Some and the kernel returns true, allocate vec![T::zero(); n], run the kernel, return Tensor::from_vector(out, dims). Otherwise fall through to the existing try_elementwise_map unchanged.
    - [ ] The four operator impls (impl Add/Sub/Mul/Div for &Tensor) currently duplicate the same broadcast logic; collapse them to call the corresponding try_* and unwrap with the same panic message. One dispatch point, no behavior change.
    - [ ] Scalar ops: same fast path inside unary_map's call sites in unary_operations.rs (out[i] = a[i] op scalar, splat-style loop), same kernel module, integer rem/div stay scalar.
    Trade-off to flag: adding ElementwiseKernel as a bound on the public arithmetic impls is a tiny, additive public API constraint. It's implemented for every built-in Element type, so no practical breakage; the alternative (runtime TypeId + transmute) is worse. Element is effectively a closed set in this crate.

- [ ] Phase 2 — std::arch kernels (after Phase 1 lands and benchmarks)
    - [ ] src/simd/x86.rs: #[target_feature(enable = "avx2")] kernels for f32/f64 add/sub/mul/div (+ FMA where profitable), dispatched via is_x86_feature_detected!("avx2") with SSE2 as the compile-time baseline on x86_64. Detection is always-on, no cargo feature.
    - [ ] src/simd/neon.rs: #[cfg(target_arch = "aarch64")] NEON kernels — compile-time guaranteed, no runtime detection needed (this dev machine benefits immediately).

- Structure kernels as chunks of lanes + scalar tail (Rust 1.90 has slice::as_chunks for clean lane handling). All unsafe confined to these two files; everything above them stays safe.
- Extend to 32-bit integer mul later if benchmarks justify it.
Reductions — strategy defined now, implemented later
- Fast path category A: full-tensor reduction (all axes) on a contiguous tensor → SIMD horizontal fold (vector accumulator over lanes + horizontal combine + scalar tail). Note: this changes float summation order → not bit-exact vs the scalar path; document it, and compare with tolerance in float tests (integer sums stay exact).
- Fast path category B: reduction over the innermost contiguous axis (e.g. axes=[last]) → row-wise horizontal folds, output order preserved.
- Everything else (non-contiguous, arbitrary axis sets) keeps the current Custom-priority iterator path unchanged.
- Plumbing: keep reduce(axes, op) as-is; reduce_sum/reduce_prod/reduce_min/reduce_max internally probe for categories A/B before falling back. reduce_mean inherits whatever reduce_sum gets.
Non-contiguous behavior (per your decision)
is_contiguous() == false (slices, permutes, broadcast views) → existing iterator path, byte-for-byte identical results to today. No hidden copies.
Testing (my QA responsibility — new tests/simd_test.rs)
- Equivalence: fast path vs independently computed expected values; and fast path vs forced-scalar path results on the same data.
- Size ladder: 0 (empty), 1, lane−1, lane, lane+1, 1M+7, plus a high-rank contiguous tensor with equal total element count (rank-independence of the fast path).
- Float specials: NaN, ±inf, −0.0, denormals — bit-exact equality with scalar (Rust doesn't enable fast-math; test guards future regressions).
- Integers: overflow-wrap consistency between kernel and scalar path in release; int div still works (fallback correctness).
- Views regression: permuted / negative-step-sliced / broadcast operands produce identical results to pre-change behavior.
- Broadcast + contiguous interaction: broadcast views (stride-0) correctly take the fallback while a truly contiguous pair takes the fast path (assert indirectly via results on sizes where bug would show).
- Full existing suite must pass untouched.
Verification
- cargo test (all suites, old + new)
- cargo bench: Phase 0 baseline vs Phase 1; contiguous large-tensor ops should approach memory-bound throughput; permuted/broadcast benches must be unchanged
- Release disassembly spot-check of the f32 add kernel confirms actual vector instructions
Risk notes
- New trait bound on public ops — mitigated by implementing it for every Element type; documented in the module.
- Output allocation is vec![T::zero(); n] + overwrite (safe, simple); no set_len/MaybeUninit games in Phase 1 — revisit only if benchmarks show the zeroing cost matters.
- Debug builds won't vectorize overflow-checked integer arithmetic — that's fine, correctness identical, perf target is --release.

**Execute in order: Phase 0 → Phase 1 → tests → verify, with Phase 2 as a follow-up step once Phase 1 numbers are in.**