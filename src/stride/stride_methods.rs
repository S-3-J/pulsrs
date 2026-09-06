/*
    Stride class acts like a navigator,
    pin pointing location of the data,
    in a vast buffer.
*/

use super::Stride;
use crate::error::PulsrsError;
use crate::shape::Shape;

impl Stride {
    pub fn from_shape(shape: &Shape) -> Self {
        if shape.is_scalar() {
            return Self { strides: vec![] };
        }

        let dims = shape.dims();
        let mut strides: Vec<usize> = vec![1; dims.len()];

        for i in (0..dims.len() - 1).rev() {
            strides[i] = dims[i + 1] * strides[i + 1];
        }

        let strides = strides.iter().map(|a| *a as isize).collect();

        Self { strides }
    }

    pub fn from_vec(strides: Vec<isize>) -> Self {
        Self {strides}
    }

    pub fn rank(&self) -> usize {
        self.strides.len()
    }

    pub fn ndim(&self) -> usize {
        self.rank()
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn get(&self, axis: usize) -> Option<isize> {
        self.strides.get(axis).copied()
    }

    pub fn offset(&self, indices: &[usize]) -> Result<isize, PulsrsError> {
        if indices.len() != self.ndim() {
            return Err(PulsrsError::RankMismatch {
                expected: self.ndim(),
                found: indices.len(),
            });
        }

        let mut final_offset: isize = 0;
                
        for (i, idx) in indices.iter().enumerate() {
            final_offset += self.get(i).unwrap() * (*idx) as isize;
        }

        Ok(final_offset)

    }
}

impl From<Vec<isize>> for Stride {
    fn from(value: Vec<isize>) -> Self {
        Stride { strides: value }
    }
}
