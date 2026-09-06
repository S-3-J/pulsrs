/*
    Shape class helps to hold shape of a buffer,
    effectively decoupling buffer and its crystallization.
*/

use super::Shape;
use std::fmt::Display;

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims, names: None }
    }

    pub fn with_names(dims: Vec<usize>, names: Vec<String>) -> Self {
        Self {
            dims,
            names: Some(names),
        }
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

    pub fn has_names(&self) -> bool {
        self.names.is_some()
    }

    pub fn iter_with_names(&self) -> Option<impl Iterator<Item = (&usize, &String)>> {
        self.names
            .as_ref()
            .map(|names| self.dims.iter().zip(names.iter()))
    }

    pub fn validate_index(&self, index: &[usize]) -> bool {
        if self.rank() != index.len() {
            return false;
        }

        for (i, j) in index.iter().zip(self.dims.iter()) 
        {
            if *i >= *j {
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // stringify, collect and join dimensions
        let final_dims = {
            if let Some(named_dims) = self.iter_with_names() {
                named_dims
                    .map(|(dim, name)| format!("{}: {}", name, dim))
                    .collect::<Vec<String>>()
                    .join(", ")
            } else {
                self.iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            }
        };

        write!(f, "({},)", final_dims)?;
        Ok(())
    }
}
