
/*
    Buffer as the name suggests, is the buffer that holds raw data. 
    It is a contiguous 1-D vector of data and is not committed to a shape.
*/

use std::ops::Index;

pub struct Buffer<T>{
    data: Vec<T>,
}

impl<T> Buffer<T>{
    
    pub fn new() -> Self {
        Self {
            data : Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data : Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn push(&mut self, value: T) {
        self.data.push(value)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> Index<usize> for Buffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}