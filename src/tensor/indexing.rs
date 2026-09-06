use std::ops::Index;

use crate::dtype::Element;

use super::{PulsrsError, Tensor};

impl<T: Element + Copy> Tensor<T> {
    pub fn get(&self, indices: &[usize]) -> Result<T, PulsrsError> {
        if self.is_scalar() {
            if !indices.is_empty() {
                return Err(PulsrsError::TensorIndexingError {
                    shape: self.shape().clone(),
                    indices: indices.to_vec(),
                });
            }

            Ok(self.buffer[self.offset])
        } else {
            if indices.len() != self.rank() {
                return Err(PulsrsError::RankMismatch {
                    expected: self.rank(),
                    found: indices.len(),
                });
            }

            for i in 0..self.rank() {
                if indices[i] >= self.shape().dims()[i] {
                    return Err(PulsrsError::TensorIndexingError {
                        shape: self.shape().clone(),
                        indices: indices.to_vec(),
                    });
                }
            }

            let offset = self.stride.offset(indices)?;

            if offset.unsigned_abs() >= self.buffer.len() {
                Err(PulsrsError::IndexOutofBounds {
                    max: self.rank(),
                    found: offset.unsigned_abs(),
                })
            } else {
                Ok(self.buffer[(self.offset as isize + offset) as usize])
            }
        }
    }

    pub(crate) fn get_from_offset(&self, offset: usize) -> Result<T, PulsrsError> {
        if self.is_scalar() {
            if offset != 0 {
                return Err(PulsrsError::IndexOutofBounds {
                    max: self.rank(),
                    found: offset,
                });
            }

            Ok(self.buffer[0])
        } else if offset >= self.buffer.len() {
            Err(PulsrsError::TensorIndexingError {
                shape: self.shape().clone(),
                indices: vec![offset],
            })
        } else {
            Ok(self.buffer[self.offset+offset])
        }        
    }
}

impl<T: Element> Index<&[usize]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        if self.is_scalar() {
            if index.is_empty() {
                return &self.buffer[0];
            } else {
                panic!(
                    "{}",
                    &PulsrsError::TensorIndexingError {
                        shape: self.shape().clone(),
                        indices: index.to_vec()
                    }
                )
            }
        }

        if self.rank() != index.len() {
            panic!(
                "{}",
                PulsrsError::TensorIndexingError {
                    shape: self.shape().clone(),
                    indices: index.to_vec()
                }
            )
        }

        for i in 0..self.rank() {
            if index[i] >= self.shape().dims()[i] {
                panic!(
                    "{}",
                    PulsrsError::TensorIndexingError {
                        shape: self.shape().clone(),
                        indices: index.to_vec()
                    }
                )
            }
        }

        let offset_wrapper = self.stride.offset(index);

        let Ok(offset) = offset_wrapper else {
            panic!("{}", offset_wrapper.unwrap_err())
        };

        &self.buffer[(self.offset as isize + offset) as usize]
    }
}
