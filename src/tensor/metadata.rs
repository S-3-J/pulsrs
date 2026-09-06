
use crate::dtype::{DType, Element};

use super::{Tensor, Shape};

impl<T: Element> Tensor<T> {
    // calculate rank
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn ndim(&self) -> usize {
        self.rank()
    }

    // shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    // strides
    pub fn strides(&self) -> &[isize] {
        self.stride.strides()
    }

    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    // scalar check
    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}