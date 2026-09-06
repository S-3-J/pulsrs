use crate::dtype::Element;

use super::{PulsrsError, Rc, Shape, Stride, Tensor};
use std::collections::HashSet;

impl<T: Element> Tensor<T>
where
    T: Copy,
{
    //permute (also transpose)
    pub fn permute(&self, permutation: Vec<usize>) -> Result<Self, PulsrsError> {
        if self.is_scalar() {
            if !permutation.is_empty() {
                return Err(PulsrsError::RankMismatch {
                    expected: 0,
                    found: permutation.len(),
                });
            }
            return Ok(self.clone());
        }

        let contiguous_flag = permutation == (0usize..self.rank()).collect::<Vec<_>>();

        if !contiguous_flag {
            let mut index_set: HashSet<usize> = HashSet::new();

            for i in &permutation {
                if i >= &self.rank() {
                    return Err(PulsrsError::AxisOutofBounds {
                        max: self.rank(),
                        min: 0,
                        found: *i,
                    });
                }
                if !index_set.insert(*i) {
                    return Err(PulsrsError::UniqueDataConstraintError { given: permutation });
                }
            }

            let curr_shape_dim = self.shape().dims();
            let curr_strides = self.strides();

            if curr_shape_dim.len() != permutation.len() {
                Err(PulsrsError::RankMismatch {
                    expected: curr_shape_dim.len(),
                    found: permutation.len(),
                })
            } else {
                let mut permuted_shape = vec![0; curr_shape_dim.len()];
                let mut permuted_stride = vec![1; curr_strides.len()];

                for i in 0..curr_shape_dim.len() {
                    permuted_shape[i] = curr_shape_dim[permutation[i]];
                    permuted_stride[i] = curr_strides[permutation[i]];
                }

                let permuted_shape: Shape = permuted_shape.into();
                let permuted_stride: Stride = permuted_stride.into();

                Ok(Tensor {
                    buffer: Rc::clone(&self.buffer),
                    shape: permuted_shape,
                    stride: permuted_stride,
                    offset: self.offset,
                    contiguous: contiguous_flag,
                    dtype: self.dtype(),
                })
            }
        } else {
            Ok(self.clone())
        }
    }
}
