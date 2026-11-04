### Python Frontend Plan for Pulsrs

## Context / Problem Statement

Pulsrs currently exposes its tensor implementation exclusively through its Rust API. The goal is to provide a Python frontend that feels familiar to users of NumPy while keeping Rust responsible for all meaningful computation.

Python should act primarily as an argument-parsing and dispatch layer. Tensor creation, arithmetic, broadcasting, reductions, slicing, reshaping, iteration over tensor elements, and other substantial work should be performed by Pulsrs in Rust.

The frontend must not silently recreate Pulsrs using NumPy, Python loops, or Python-side tensor arrays.

---

## Decisions

- Binding layer: PyO3
- Packaging/build tool: maturin
- Supported Python versions: Python 3.11+
- Python semantics: NumPy-like where practical
- Repository layout: Cargo workspace containing the core library and Python bindings
- Python package name: `pulsrs`
- Computation location: Rust
- Python responsibility:
  - API ergonomics
  - Argument parsing and normalization
  - Dispatching to Rust
  - Converting Python objects into Rust-compatible inputs
  - Converting Rust results back into Python objects
- Rust responsibility:
  - Tensor storage and metadata
  - Arithmetic and reductions
  - Broadcasting
  - Indexing and slicing
  - Reshaping and view creation
  - Contiguity and copy decisions
  - All element-level computation

---

## Relationship to the SIMD Plan

The Python frontend must not depend on architecture-specific SIMD behavior.

SIMD is an internal Rust optimization. The Python API must produce the same observable results regardless of whether an operation uses:

- a SIMD kernel,
- the scalar fallback,
- a contiguous fast path,
- or the general strided iterator path.

The current SIMD Phase 1 work is still relevant to the Python frontend because the frontend may receive contiguous tensors that are views into larger allocations. Before SIMD Phase 1 is sealed, Pulsrs must provide an offset-aware method for obtaining the logical contiguous slice represented by a tensor.

The Python frontend should use the same guarded contiguous-slice mechanism once it exists. It must not independently inspect private buffer fields or assume that the beginning of the backing buffer is the beginning of the tensor's logical data.

---

## Goals

- Provide a `pulsrs.Tensor` Python class.
- Support Python 3.11 and newer.
- Provide familiar NumPy-like construction, metadata, arithmetic, indexing, slicing, reshaping, and reduction APIs.
- Ensure all tensor computation is performed in Rust.
- Preserve Pulsrs view semantics where possible.
- Expose tensor data to NumPy without unnecessary copying when safe.
- Convert Rust errors into appropriate Python exceptions.
- Ensure Rust panics do not cross the Python boundary.
- Provide reproducible local builds and distributable wheels.
- Keep the Python frontend independent of the internal SIMD implementation.

---

## Non-Goals for the Initial Release

- GPU support
- Autograd
- JIT compilation
- Python callbacks invoked for every tensor element
- Full NumPy API compatibility
- Advanced matrix multiplication APIs
- Complex-number support
- float16 and bfloat16 support
- i128 and u128 buffer interoperability
- Full support for i128 and u128 Python tensors
- Thread-safe mutation of tensors
- Arbitrary in-place mutation semantics
- Removing the Python GIL
- Refactoring every Rust panic-based API
- Exposing every Rust module directly to Python

---

## Proposed Repository Layout

```text
pulsars/
├── Cargo.toml
├── Cargo.lock
├── crates/
│   ├── pulsrs-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   └── pulsrs-python/
│       ├── Cargo.toml
│       └── src/
├── python/
│   ├── pulsrs/
│   │   ├── __init__.py
│   │   ├── __init__.pyi
│   │   ├── _core.pyi
│   │   └── py.typed
│   └── tests/
├── pyproject.toml
├── maturin.toml
└── README.md
The existing library may initially remain at the workspace root to minimize migration churn. A later reorganization can move it into crates/pulsrs-core once the binding crate is stable.
The workspace should contain:
- The core Pulsrs crate
- The PyO3 binding crate
- Existing Rust tests and benchmarks
- Python tests
- Packaging configuration
The core crate must not depend on PyO3. The Python binding crate must depend on the core crate.
Python Tensor Representation
Pulsrs's Rust Tensor<T> is generic over its element type. Python cannot directly represent a generic Rust type, so the binding should use a non-generic Python class backed by a non-generic Rust enum.
enum TensorValue {
    F32(Tensor<f32>),
    F64(Tensor<f64>),
    I8(Tensor<i8>),
    I16(Tensor<i16>),
    I32(Tensor<i32>),
    I64(Tensor<i64>),
    U8(Tensor<u8>),
    U16(Tensor<u16>),
    U32(Tensor<u32>),
    U64(Tensor<u64>),
    Bool(Tensor<bool>),
}
A macro should generate repetitive dispatch code for:
- Construction
- Metadata
- Arithmetic
- Comparison
- Reduction
- Indexing
- Slicing
- Conversion
- Contiguous-buffer export
The Python-visible class should remain a single type:
pulsrs.Tensor
Differences in Rust element type must not result in different Python classes.
Initial dtype set
- float32
- float64
- int8
- int16
- int32
- int64
- uint8
- uint16
- uint32
- uint64
- bool
i128 and u128 should be excluded from the initial Python API because they cannot use Python's standard fixed-width buffer protocol.
Scalar and zero-dimensional tensors
A scalar tensor must have:
- shape == ()
- ndim == 0
- size == 1
- is_scalar == True
A Python scalar and a zero-dimensional tensor should not be silently conflated. Operations should follow this rule:
- tensor + 2 returns a tensor.
- tensor[()] returns a tensor view when supported.
- tensor.item() returns a Python scalar.
Python Package Structure
The public package should contain only curated exports:
from pulsrs import Tensor
from pulsrs import array, asarray, zeros, ones, full, arange, linspace
Recommended organization:
python/pulsrs/
├── __init__.py       # Public exports
├── __init__.pyi      # Public type stubs
├── _core.pyi         # Extension-module type stubs
└── py.typed          # Marks the package as typed
All tensor implementation details should remain in the native extension.
Python wrappers should be used only where they provide meaningful ergonomics, such as:
- normalizing nested input sequences,
- translating Python indexing expressions,
- normalizing slice bounds,
- constructing clear exception messages.
They should not traverse tensor elements to perform computation.
Initial Python API
Construction
pulsrs.array(object, dtype=None, copy=True)
pulsrs.asarray(object, dtype=None)
pulsrs.tensor(object, dtype=None)
pulsrs.zeros(shape, dtype="float32")
pulsrs.ones(shape, dtype="float32")
pulsrs.full(shape, fill_value, dtype=None)
pulsrs.empty(shape, dtype="float32")
pulsrs.arange(start, stop=None, step=1, dtype=None)
pulsrs.linspace(start, stop, num=50, dtype=None)
Construction requirements:
- Nested Python sequences must be rectangular.
- Ragged sequences must raise ValueError.
- An empty sequence must have an explicitly defined shape rather than silently selecting an arbitrary rank.
- Scalar inputs must create zero-dimensional tensors.
- Shape validation must occur in Rust.
- Data copying should occur in Rust where practical.
- Python should not flatten a large nested sequence using a Python element loop.
empty may reuse Pulsrs's existing zero or uninitialized allocation behavior only after the core provides a clearly defined API. Until then, it should remain out of the initial release.
Metadata
tensor.shape
tensor.dtype
tensor.ndim
tensor.size
tensor.itemsize
tensor.is_contiguous
tensor.is_scalar
tensor.strides
tensor.offset
Metadata should come directly from the Rust tensor rather than being recomputed in Python.
Conversion
tensor.item()
tensor.tolist()
tensor.to_numpy()
numpy.asarray(tensor)
Conversion semantics:
- item() requires exactly one element.
- tolist() returns nested Python lists matching the tensor shape.
- to_numpy() returns a NumPy array.
- A contiguous tensor should be exported without copying when the Python buffer protocol can safely represent its data.
- A non-contiguous tensor should be materialized into a contiguous copy.
- The exported buffer must remain valid only while its owning Python tensor is alive.
- The extension must not expose mutable access to memory that Pulsrs considers immutable.
Arithmetic
left + right
left - right
left * right
left / right
left == right
left != right
left < right
left <= right
left > right
left >= right
Arithmetic requirements:
- Tensor-tensor operations dispatch to Rust.
- Tensor-scalar operations dispatch to Rust.
- Broadcasting follows NumPy's trailing-dimension rules.
- Shape incompatibility raises ValueError.
- Mixed dtype operations follow an explicit Rust-defined promotion policy.
- Unsupported promotion combinations raise TypeError.
- Python must not materialize tensors into NumPy arrays to perform arithmetic.
- Division by zero follows the selected Rust semantics and must be tested and documented.
Unary operations
-x
+x
abs(x)
Initial support should include negation and absolute value if corresponding safe Rust APIs are available. Other transcendental operations should be added only when the Rust core exposes them.
Reductions
tensor.sum(axis=None, keepdims=False)
tensor.prod(axis=None, keepdims=False)
tensor.min(axis=None, keepdims=False)
tensor.max(axis=None, keepdims=False)
tensor.mean(axis=None, keepdims=False)
Reduction requirements:
- Reductions must execute in Rust.
- Axis handling must follow NumPy conventions.
- Repeated axes and invalid axes must raise ValueError.
- keepdims=True must preserve reduced dimensions with size one.
- Full reductions must follow the selected axis-ordering contract.
- Floating-point reductions must be tested using tolerances where summation order may differ.
- Integer reductions must remain exact.
The Rust core may need explicit fallible reduction APIs with axis and keepdims parameters. The Python binding should not emulate these semantics by repeatedly slicing and reducing tensors in Python.
Shape and view operations
tensor.reshape(shape)
tensor.flatten()
tensor.permute(order)
tensor.transpose(axes=None)
tensor.contiguous()
tensor.copy()
Semantics:
- reshape follows NumPy behavior where possible.
- A contiguous reshape should remain a view.
- A non-contiguous reshape may copy, matching Pulsrs semantics.
- transpose defaults to reversing dimensions.
- permute accepts an explicit dimension order.
- contiguous explicitly materializes a contiguous copy.
- copy always returns storage independent of the source.
- View-producing operations must preserve shape, stride, offset, and contiguity invariants.
Indexing
Python indexing should support:
- A single integer index
- A tuple of integer indices
- A tuple containing slices
- Negative integer indices
- Negative slice bounds
- Negative slice steps
- None as a new axis
- Ellipsis where supported
- Out-of-bounds error reporting
Recommended behavior:
tensor[0]
tensor[-1]
tensor[1, 2]
tensor[0, ...]
tensor[None, :, 2]
tensor[1:5:2]
Requirements:
- Integer indexing should return a tensor view where Pulsrs supports it.
- Partial indexing should preserve the unselected dimensions.
- Invalid indices raise IndexError.
- Invalid rank combinations raise IndexError or ValueError consistently.
- Negative indices must be normalized before reaching Rust.
- Python must not perform element-by-element lookup for supported tensor indexing.
Slicing
Requirements:
- Slice bounds must follow Python's half-open interval semantics.
- Negative bounds must be normalized relative to dimension length.
- Negative steps must use Pulsrs's signed stride support.
- None must insert a new axis without consuming a source axis.
- Slices should be views whenever possible.
- The Python binding must not copy a sliced tensor merely to implement Python's indexing syntax.
The current Rust slicing API does not fully match these semantics. The core should therefore provide a normalized, fallible slicing contract that the binding can call directly.
Broadcasting
tensor.broadcast(shape)
Requirements:
- Compatibility must be validated before creating a broadcast view.
- Incompatible shapes raise ValueError.
- Broadcasting must create a zero-stride view where applicable.
- Binary-operation broadcasting must use the same Rust compatibility rules.
- The Python binding should not manually insert dimensions for every operation.
Dtype Promotion Policy
The current Rust tensor type fixes its element type statically. The initial binding should not use unchecked runtime casts.
The Rust core should define a promotion policy for supported dtypes.
A possible initial policy is:
- Same dtype: preserve dtype.
- Integer and floating-point: promote to the floating-point dtype.
- Signed and unsigned integers: promote to a signed integer dtype that safely represents both.
- Boolean with integer: promote to an integer dtype.
- Boolean with floating-point: promote to the floating-point dtype.
- Float32 with Float64: promote to Float64.
Before implementation, the exact matrix should be documented and tested. Any combination that cannot be represented safely should raise TypeError.
Promotion must happen in Rust. Python should not convert buffers itself to force a result dtype.
Error Mapping
The binding must call fallible Rust APIs wherever possible and must not intentionally expose panic-based Rust APIs.
Recommended Python exception mapping:
Rust condition	Python exception
Invalid or out-of-bounds index	IndexError
Invalid shape or reshape	ValueError
Incompatible broadcasting shapes	ValueError
Invalid reduction axis	ValueError
Unsupported dtype combination	TypeError
Invalid argument type	TypeError
Invalid Python sequence	ValueError
Internal allocation or memory failure	MemoryError
The binding should not rely on parsing Rust error display strings. It should match the error variants directly.
Rust panics should be contained and converted into a Python-facing internal error where PyO3 provides a safe mechanism. A panic must never unwind into the Python interpreter as an uncontrolled crash.
Zero-Copy and View Semantics
Pulsrs uses shared storage for views. The Python frontend must preserve this distinction.
A view-producing operation should:
- Share storage with its source.
- Reflect any future mutation only if the core later introduces supported mutable semantics.
- Preserve logical shape and strides.
- Preserve the tensor's backing-buffer offset.
A copy-producing operation should:
- Have storage independent from its source.
- Be contiguous when documented to be contiguous.
- Not be affected by changes to the source.
The initial frontend is expected to be functionally immutable because the current public Rust tensor API does not provide normal element mutation.
Python's buffer protocol must not accidentally provide mutation of Pulsrs-owned memory. If a mutable buffer is exposed, the core must explicitly define and test the corresponding mutation semantics.
NumPy Interoperability
NumPy is an interoperability target, not the computation backend.
Pulsrs operations must never be implemented as:
return np.asarray(self) + np.asarray(other)
Interoperability should include:
- numpy.asarray(tensor)
- tensor.to_numpy()
- Correct dtype mapping
- Correct shape and dimension ordering
- Correct handling of contiguous tensors
- Correct materialization of non-contiguous tensors
- Correct lifetime behavior for exported buffers
Initial dtype mapping:
Pulsrs	NumPy
float32	numpy.float32
float64	numpy.float64
int8	numpy.int8
int16	numpy.int16
int32	numpy.int32
int64	numpy.int64
uint8	numpy.uint8
uint16	numpy.uint16
uint32	numpy.uint32
uint64	numpy.uint64
bool	numpy.bool_
NumPy should be a test-only optional dependency unless it is required by a public interoperability feature.
Threading and the GIL
The current Pulsrs tensor storage uses Rc, making it neither Send nor Sync.
The initial binding should therefore:
- Keep the GIL during calls that access tensor storage.
- Mark the Python tensor wrapper as non-sendable if required by PyO3.
- Avoid exposing thread-safe mutation.
- Avoid claiming that tensor operations can safely run concurrently.
Removing the GIL requires a separate core design change, potentially replacing Rc with Arc or another ownership model. That work should not be mixed into the initial frontend.
Core Rust API Preparation
The Python binding should adapt a deliberately curated Rust API rather than mirror the entire crate.
Offset-aware contiguous data
Pulsrs needs a safe method that returns the logical contiguous slice for a tensor:
pub fn as_contiguous_slice(&self) -> Option<&[T]>
The method must:
- Return None for non-contiguous tensors.
- Start at the tensor's logical offset.
- Cover exactly numel() elements.
- Verify that the requested range is within the backing buffer.
- Avoid exposing unrelated prefix or suffix elements.
- Work for contiguous views into larger allocations.
This method should serve both SIMD kernels and Python buffer export.
Safe data export
The binding may also need a method returning the contiguous storage without copying. It should reuse the same invariant checks as as_contiguous_slice.
Fallible shape operations
Add or expose fallible APIs for:
- Reshaping
- Flattening
- Permuting
- Transposing
- Broadcasting
- Contiguous materialization
Python must not rely on Rust's panic-based operator implementations.
Normalized slicing
The core should provide a fallible slicing operation that understands:
- Resolved dimension ranges
- Negative steps
- Integer-index removal
- New-axis insertion
- Ellipsis expansion, if Ellipsis is supported
The binding should normalize Python syntax and delegate the actual traversal/view creation to Rust.
Explicit reductions
The core should provide an API capable of expressing:
- Reduced axes
- Repeated-axis validation
- keepdims
- Full reduction
- Partial reduction
Negation and absolute value
Add safe, fallible unary operations where they do not already exist.
Promotion
Define promotion as a Rust-level operation. The binding should dispatch to it rather than reproducing the matrix in Python.
Packaging and Build Configuration
Use pyproject.toml with maturin as the build backend.
Example structure:
[build-system]
requires = ["maturin>=1.9,<2.0"]
build-backend = "maturin"

[project]
name = "pulsrs"
requires-python = ">=3.11"
dynamic = ["version"]

[tool.maturin]
bindings = "pyo3"
python-source = "python"
module-name = "pulsrs._core"
Requirements:
- Build against the abi3-py311 feature where PyO3 support is stable.
- Keep the core crate free of Python-specific dependencies.
- Include required Rust sources in source distributions.
- Keep Python package version and Rust crate version synchronized.
- Generate or maintain type stubs.
- Mark the package as typed using py.typed.
Implementation Phases
Phase 0 — API Contract and Baseline
- Confirm the initial supported dtype set.
- Define the dtype promotion matrix.
- Define error-to-exception mapping.
- Define construction semantics for empty and ragged sequences.
- Define indexing and slicing semantics.
- Define reduction-axis and keepdims semantics.
- Define view and copy semantics.
- Define NumPy interoperability guarantees.
- Record Rust benchmark baselines before frontend development.
Phase 1 — Workspace and Package Scaffold
- Convert the repository into a Cargo workspace.
- Add the pulsrs-python PyO3 crate.
- Add pyproject.toml and maturin configuration.
- Create the python/pulsrs package.
- Register and initialize the native module.
- Export an empty Tensor class.
- Build, install, and import the package locally.
- Add a minimal packaging smoke test.
Phase 2 — Core API Hardening
- Implement the offset-aware as_contiguous_slice contract.
- Add safe contiguous storage export.
- Add fallible reshape, permute, transpose, and broadcast paths.
- Add the normalized fallible slicing contract.
- Add explicit reduction-axis APIs.
- Add safe unary operations required by Python.
- Define and implement Rust dtype promotion.
- Add Rust regression tests for each new core contract.
- Confirm the existing Rust suite still passes unchanged.
Phase 3 — Python Tensor Wrapper
- Implement the non-generic Tensor PyO3 class.
- Implement the internal TensorValue enum.
- Generate dtype dispatch through a macro.
- Implement metadata accessors.
- Implement scalar-tensor semantics.
- Implement repr and str.
- Map Rust errors to Python exceptions.
- Prevent Rust panics from crossing the boundary.
- Add __array__ or the selected NumPy conversion protocol.
Phase 4 — Construction and Conversion
- Implement array, asarray, and tensor.
- Implement zeros, ones, and full.
- Implement arange and linspace.
- Implement sequence-to-buffer conversion in Rust.
- Reject ragged sequences with ValueError.
- Implement item and tolist.
- Implement to_numpy.
- Implement zero-copy buffer export for eligible contiguous tensors.
- Materialize non-contiguous tensors for NumPy export.
- Add Python type stubs.
Phase 5 — Arithmetic, Comparison, and Reductions
- Bind binary tensor-tensor arithmetic.
- Bind tensor-scalar arithmetic.
- Implement Python operator overloads.
- Implement comparison operations.
- Implement Rust-backed broadcasting.
- Implement dtype promotion.
- Bind reductions.
- Implement axis and keepdims.
- Bind negation and absolute value.
- Ensure no arithmetic uses a Python element loop or NumPy backend.
Phase 6 — Indexing, Slicing, and Views
- Implement integer indexing.
- Implement tuple indexing.
- Implement negative-index normalization.
- Implement slice normalization.
- Implement negative-step slicing.
- Implement new-axis insertion.
- Implement Ellipsis handling if supported by the core.
- Implement reshape, flatten, permute, and transpose.
- Implement broadcast, contiguous, and copy.
- Verify view aliasing and copy independence.
- Verify offsets, strides, and contiguity after chained operations.
Phase 7 — Performance and Concurrency Evaluation
- Benchmark Python-to-Rust call overhead for small tensors.
- Benchmark large-tensor arithmetic through Python.
- Compare Python-facing performance with direct Rust benchmarks.
- Determine whether GIL release is worth the Rc to Arc core migration.
- Evaluate zero-copy conversion overhead for common NumPy workflows.
- Avoid premature core refactoring based on microbenchmarks alone.
Phase 8 — Packaging, CI, and Release
- Build CPython 3.11+ wheels.
- Build an abi3-py311 wheel where practical.
- Add Linux CI.
- Add macOS CI for supported architectures.
- Add Windows CI if the core and packaging support it.
- Build and test source distributions.
- Test installation in a clean environment.
- Publish documentation for supported dtypes and operations.
- Document NumPy interoperability and view semantics.
- Document the current GIL behavior.
Testing Strategy
Rust Core Tests
Add Rust tests before relying on the corresponding Python behavior:
- Contiguous tensor with nonzero offset
- Contiguous view into a larger buffer
- Non-contiguous tensor with nonzero offset
- Empty tensors
- Scalar tensors
- Broadcasting compatibility and rejection
- Reshape and permute metadata invariants
- Negative-step slicing
- New-axis insertion
- Reduction axes and keepdims
- Dtype promotion matrix
- Divided-by-zero behavior
- Integer overflow behavior in debug and release builds
Python API Tests
Use pytest for:
- Package import and public exports
- Constructor behavior
- Nested, empty, scalar, and ragged input
- Shape, dtype, rank, size, and contiguity metadata
- Scalar and zero-dimensional tensor behavior
- Tensor-tensor and tensor-scalar arithmetic
- Mixed-dtype arithmetic and promotion
- Broadcasting and incompatible shapes
- Comparison operations
- Reductions with axes and keepdims
- Integer, slice, tuple, negative, and Ellipsis indexing
- Negative slice steps
- New-axis insertion
- Reshape, permute, transpose, contiguous, and copy
- View aliasing and copy independence
- item, tolist, and to_numpy
- Buffer-protocol lifetime and immutability
- Exception types
- Empty dimensions and zero-element reductions
- Float specials, including NaN and signed zero
- Integer overflow behavior
- Division by zero
- Garbage collection and tensor lifetime
- Chained Python-side operations
Differential Tests
NumPy may be used as a test oracle for operations whose semantics are intentionally NumPy-like.
These tests must not influence the implementation architecture. The actual operation must still run in Rust.
Differential testing should cover:
- Shapes
- Broadcasting
- Arithmetic
- Comparisons
- Reductions
- Indexing
- Slicing
- Reshape and transpose
- NumPy conversion
Use exact comparison for integers and metadata. Use documented tolerances for floating-point reductions where summation order can differ.
Adversarial Tests
Include:
- Scalar operands
- Empty tensors
- Zero-length dimensions
- Singleton dimensions
- High-rank tensors
- Very large tensors
- Mixed contiguous and non-contiguous operands
- Broadcast operands with zero strides
- Repeated chained views
- Negative steps followed by transpose
- Non-zero-offset views
- Scalar extraction from tensors with one element
- Reduction over empty axes
- Duplicate reduction axes
- Unsupported dtype combinations
- Exception behavior for invalid indices and shapes
Verification Commands
Rust:
cargo fmt --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo test --workspace --release
cargo bench --workspace
Python:
python -m pytest
python -m pytest --strict-markers
Package verification:
maturin build
maturin develop
python -c "import pulsrs; print(pulsrs.__version__)"
A clean-environment installation test should build a wheel, install it into a fresh virtual environment, and run the Python test suite against the installed wheel rather than the repository checkout.
Risks and Mitigations
Rust core uses Rc
Rc prevents Send and Sync, which constrains PyO3 concurrency and GIL release.
Mitigation: keep the GIL initially and defer the ownership-model migration until benchmarks justify it.
Generic Rust tensor types
Tensor<T> cannot map directly to one Python class.
Mitigation: use a non-generic internal enum and macro-generated dispatch.
Contiguous tensors can be views
A contiguous view may begin at a nonzero offset and occupy only part of its backing allocation.
Mitigation: use one offset-aware contiguous-slice contract for SIMD and Python buffer export.
Python slicing differs from Rust slicing
The current Rust slicing API does not fully implement Python's new-axis, negative-index, or partial-index semantics.
Mitigation: add a normalized fallible core slicing API rather than duplicating tensor traversal in Python.
Several Rust APIs panic
Panic-based APIs are unsafe to expose across the Python boundary.
Mitigation: expose fallible wrappers and convert Rust errors into Python exceptions.
Dtype promotion is absent
NumPy-like mixed-dtype operations require explicit promotion rules.
Mitigation: define promotion in Rust, test the complete matrix, and reject unsafe casts.
Buffer export can leak mutability or invalid lifetimes
Exporting tensor memory through Python requires careful lifetime and immutability handling.
Mitigation: export only validated logical contiguous storage and keep ownership tied to the Python tensor.
Non-contiguous conversion requires copying
NumPy interoperability may need a copy for non-contiguous views.
Mitigation: make this behavior explicit and test it. Do not silently claim that every conversion is zero-copy.
Python overhead dominates small operations
Cross-language call overhead may dominate small tensors.
Mitigation: benchmark small and large tensors separately. Optimize dispatch later rather than prematurely complicating the core.
Execution Order
1. Phase 0 — API contract and baseline
2. Phase 1 — Workspace and package scaffold
3. Phase 2 — Core API hardening
4. Phase 3 — Python tensor wrapper
5. Phase 4 — Construction and conversion
6. Phase 5 — Arithmetic, comparison, and reductions
7. Phase 6 — Indexing, slicing, and views
8. Phase 7 — Performance and concurrency evaluation
9. Phase 8 — Packaging, CI, and release
The core API contracts required by the binding should be completed before exposing the corresponding Python functionality. The frontend should expose only operations whose Rust semantics and error behavior are already defined and tested.