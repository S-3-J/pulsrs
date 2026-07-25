
/*
    Shape class helps to hold shape of a buffer,
    effectively decoupling buffer and its crystallization.
*/

use std::fmt::Display;
use super::Shape;

impl Shape {
    
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn scalar() -> Self {
        Self::new(vec![])
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn ndim(&self) -> usize {
        self.rank()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    pub fn get(&self, axis: usize) -> Option<usize> {
        self.dims.get(axis).copied()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.dims.iter()
    }

    pub fn validate_index(&self, index: &[usize]) -> bool {
        if self.rank() != index.len() {
            return false;
        }

        for i in 0..self.rank() {
            if index[i] >= self.dims[i] {
                return false;
            }
        }
        
        true
    }
}


impl Default for Shape {
    fn default() -> Self {
        Self::scalar()
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self::new(value)
    }
}

impl Display for Shape {
    fn fmt(
        &self, 
        f: &mut std::fmt::Formatter<'_>
    ) -> std::fmt::Result {
        
        // stringify, collect and join dimensions
        let dims: String = self.iter().map(|n| n.to_string()).collect::<Vec<String>>().join(", ");

        write!(
            f,
            "({},)",
            dims
        )
    }
}