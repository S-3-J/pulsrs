# Tensor Core Contract

Status: **draft baseline contract; semantic review still required**

This document records the behavior the current implementation exposes or intends to preserve. It is not a claim of full PyTorch compatibility.

## Storage and metadata

- A tensor stores a one-dimensional contiguous buffer plus logical shape, strides, offset, contiguity, and dtype metadata.
- Shape dimensions are ordered and row-major contiguous strides are derived from the shape.
- A scalar has rank zero, one stored element, and no dimensions/strides.
- A dimension of zero produces an empty tensor with zero logical elements.
- Logical indexing resolves through strides and offset; storage is not itself multidimensional.

## Views and copies

- `permute`, `slice`, and `broadcast` are intended to be zero-copy views sharing the source buffer.
- `reshape`/`view` preserve the existing storage when the current layout permits it; otherwise the implementation may materialize or reject according to the current operation contract.
- `contiguous()` returns an equivalent row-major tensor and materializes non-contiguous layouts.
- A view must preserve logical element order even when its storage is non-contiguous.
- A zero-stride broadcast dimension repeats a source value without replicating storage.

## Operations

- Elementwise arithmetic requires compatible shapes or uses supported broadcasting.
- `try_*` and `try_elementwise_map` are fallible APIs for shape and mapping failures.
- Operator overloads are convenience APIs and may panic when their fallible counterpart fails.
- Reductions accept explicit axis lists and currently support sum, product, min, max, and mean.
- Iteration must produce exactly `numel()` logical values in the requested traversal order.

## Dtypes

- Built-in signed/unsigned integers, `f32`, `f64`, and `bool` implement the current element foundation.
- Each tensor carries a `DType` tag matching its element type.
- Cross-dtype promotion, conversion, mixed-dtype arithmetic, and complex types are not part of the current contract.

## Errors and invalid input

- Fallible constructors and view operations return typed `PulsrsError` variants for documented failures.
- Indexing through Rust’s `Index` trait is an infallible convenience API and currently panics for invalid rank/index input.
- Reduction axis errors are not yet uniformly represented across all convenience methods.
- Overflow, empty-reduction behavior, and some invalid slice/range cases require explicit semantic decisions before Phase 1 closes.

## Compatibility boundary

The target is semantic compatibility with important tensor behavior, not strict PyTorch API parity. Unsupported behavior must be documented rather than silently approximated.

## Open decisions before Phase 1

1. Exact behavior for reductions over empty tensors.
2. Whether `reshape` may copy or must reject non-contiguous inputs.
3. Whether broadcast shape validation belongs in `broadcast()` or only in higher-level operators.
4. Dtype promotion and mixed-type operation rules.
5. Mutation policy once autograd and thread-safe execution are introduced.
6. Which convenience methods must become fallible before the public backend contract is frozen.
