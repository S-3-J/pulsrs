
use super::{Tensor, Shape};

impl<T> Tensor<T> {
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
    pub fn strides(&self) -> &[usize] {
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
}