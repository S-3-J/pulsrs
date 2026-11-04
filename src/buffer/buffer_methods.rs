use super::Buffer;
use std::ops::{Index, IndexMut};

impl<T> Buffer<T> {
    /// Creates a new empty Buffer.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates a new empty Buffer, already allocating `capacity: usize` space in memory.
    /// Insertions upto `capacity` elements do not perform costly copy and allocations.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Returns the current length of the Buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the current capacity of the Buffer.
    /// Capacity means, memory pre-reserved by the Buffer.
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Returns true is Buffer's `length` is 0.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Pushes a value into a Buffer, in other words, it appends a value at the last of the Buffer.
    /// Requires a mutable reference to the Buffer object.
    pub fn push(&mut self, value: T) {
        self.data.push(value)
    }

    /// Returns an immutable slice of the contiguous data stored inside.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice of the contiguous data stored inside.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a Buffer, containing same elements and capacity as the given `data: Vec<T>` argument.
    /// Consumes `data: Vec<T>`
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Return a rust Vector, containing the data held by the buffer.
    /// Consumes self.
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

impl<T> IndexMut<usize> for Buffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> Default for Buffer<T> {
    fn default() -> Self {
        Self::new()
    }
}
