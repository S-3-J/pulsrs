use std::fmt::Debug;

use crate::{
    dtype::Element,
    error::{MemoryError, TensorError},
    function_signature,
};

use super::{PulsrsError, Shape, Stride, Tensor};

impl<T: Element> Tensor<T>
where
    T: Copy + Default + Debug,
{
    pub fn view(&self, dims: Vec<usize>) -> Result<Self, PulsrsError> {
        let new_shape: Shape = dims.into();

        if self.shape.numel() != new_shape.numel() {
            Err(TensorError::ReshapedElementMismatch {
                expected: self.shape.numel(),
                found: new_shape.numel(),
            }
            .into())
        } else {
            let new_strides = Stride::from_shape(&new_shape);
            if !self.is_contiguous() {
                Err(MemoryError::NonContiguousMemoryError {
                    function_path: function_signature!(),
                }
                .into())
            } else {
                let mut t = self.clone();
                t.shape = new_shape;
                t.stride = new_strides;

                Ok(t)
            }
        }
    }

    pub fn reshape(&self, dims: Vec<usize>) -> Result<Self, PulsrsError> {
        let new_shape: Shape = dims.into();

        if self.shape.numel() != new_shape.numel() {
            Err(TensorError::ReshapedElementMismatch {
                expected: self.shape.numel(),
                found: new_shape.numel(),
            }
            .into())
        } else {
            let new_strides = Stride::from_shape(&new_shape);
            let mut t = self.contiguous();

            t.shape = new_shape;
            t.stride = new_strides;

            Ok(t)
        }
    }

    pub fn flatten(&self) -> Result<Self, PulsrsError> {
        let numel = self.numel();
        self.reshape(vec![numel])
    }
}
