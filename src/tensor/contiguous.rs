use std::fmt::Debug;

use crate::{
    dtype::Element,
    error::{MemoryError, PulsrsError},
    function_signature,
    iterator::iteration_style::IterStyle::Cstyle,
};

use super::{Buffer, Rc, Stride, Tensor};

impl<T: Element> Tensor<T>
where
    T: Copy + Default + Debug,
{
    pub fn contiguous(&self) -> Self {
        if self.is_scalar() {
            return self.clone();
        }

        if self.is_contiguous() {
            self.clone()
        } else {
            let mut contiguous_buffer: Vec<T> = Vec::with_capacity(self.shape().numel());

            let new_stride = Stride::from_shape(&self.shape);

            let tensor_iterator = self.tensor_iterator(Cstyle);

            for x in tensor_iterator {
                contiguous_buffer.push(x);
            }

            Tensor {
                buffer: Rc::new(Buffer::from_vec(contiguous_buffer)),
                shape: self.shape.clone(),
                stride: new_stride,
                offset: 0,
                contiguous: true,
                dtype: self.dtype(),
            }
        }
    }

    pub fn buffer_as_slice(&self) -> Result<&[T], PulsrsError> {
        if !self.is_contiguous() {
            Err(MemoryError::NonContiguousMemoryError {
                function_path: function_signature!(),
            }
            .into())
        } else {
            let final_offset = self.offset + self.numel();
            let buffer = &self.buffer.as_slice()[self.offset..final_offset];
            Ok(buffer)
        }
    }
}
