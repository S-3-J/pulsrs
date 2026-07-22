
/*
    Stride class acts like a navigator,
    pin pointing location of the data,
    in a vast buffer.
*/

use crate::error::error::PulsrsError;
use crate::shape::shape::Shape;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stride {
    strides : Vec<usize>,
}

impl Stride {
    pub fn from_shape(shape : &Shape) -> Self {

        if shape.is_scalar() {
            return Self { strides: vec![], };
        }

        let dims = shape.dims();
        let mut strides : Vec<usize> = vec![1; dims.len()];


        for i in (0..dims.len()-1).rev() {
            strides[i] = dims[i+1] * strides[i+1];
        }

        Self { strides }
    }

    pub fn rank(&self) -> usize {
        self.strides.len()
    }

    pub fn ndim(&self) -> usize {
        self.rank()
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn get(&self, axis : usize) -> Option<usize> {
        self.strides.get(axis).copied()
    }

    pub fn offset(&self, indices: &[usize]) -> Result<usize, PulsrsError> {
        
        if indices.len() != self.ndim() {
            return Err(PulsrsError::RankMismatch{
                expected: self.ndim(),
                found: indices.len(),
            });
        }

        let mut final_offset: usize = 0;

        for i in 0..self.rank() {
            final_offset += self.get(i).unwrap() * indices[i];
        }

        Ok(final_offset)
    }
}

impl From<Vec<usize>> for Stride {
    fn from(value: Vec<usize>) -> Self {
        Stride {
            strides: value,
        }
    }
}