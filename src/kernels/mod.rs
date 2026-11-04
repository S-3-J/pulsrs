// add kernels for operations
use crate::simd::Vectorizable;

// LLVM Autovectorizable operation executor
#[allow(dead_code)]
pub(crate) struct BinarySimdKernel<T: Vectorizable, K: Vectorizable> {
    in_val: T,
    out_val: K,
}

impl<T: Vectorizable, K: Vectorizable> BinarySimdKernel<T, K> {
    pub fn try_new() -> Self {
        Self {
            in_val: T::one(),
            out_val: K::one(),
        }
    }

    pub fn operate(&self, buf_a: &[T], buf_b: &[T], out_buf: &mut [K], op: impl Fn(T, T) -> K) {
        // LLVM at ease.
        assert!(buf_a.len() == buf_b.len());

        for ((&a, &b), o) in buf_a.iter().zip(buf_b.iter()).zip(out_buf.iter_mut()) {
            *o = op(a, b);
        }
    }
}

#[allow(dead_code)]
pub(crate) struct UnarySimdKernel<T: Vectorizable, K: Vectorizable> {
    in_val: T,
    out_val: K,
}

impl<T: Vectorizable, K: Vectorizable> UnarySimdKernel<T, K> {
    pub fn try_new() -> Self {
        Self {
            in_val: T::one(),
            out_val: K::one(),
        }
    }

    pub fn operate(&self, buf: &[T], x: T, out_buf: &mut [K], op: impl Fn(T, T) -> K) {
        assert!(buf.len() <= out_buf.len());

        for (&a, o) in buf.iter().zip(out_buf.iter_mut()) {
            *o = op(a, x);
        }
    }
}

#[allow(dead_code)]
pub(crate) struct FunctionSimdKernel<T: Vectorizable, K: Vectorizable> {
    in_val: T,
    out_val: K,
}

impl<T: Vectorizable, K: Vectorizable> FunctionSimdKernel<T, K> {
    pub fn try_new() -> Self {
        Self {
            in_val: T::one(),
            out_val: K::one(),
        }
    }

    pub fn operate(&self, buf: &[T], out_buf: &mut [K], op: impl Fn(T) -> K) {
        assert!(buf.len() <= out_buf.len());

        for (&a, o) in buf.iter().zip(out_buf.iter_mut()) {
            *o = op(a);
        }
    }
}
