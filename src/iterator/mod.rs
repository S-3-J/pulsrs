
use crate::iterator::iterator_backend::IteratorBackend;
#[allow(unused_imports)]
use crate::{tensor::Tensor, error::PulsrsError, cursor::IndexCursor};

pub mod indexiterator;
pub mod tensoriterator;
pub mod iteration_style;
pub mod iterator_backend;

pub struct IndexIterator {
    cursor: IndexCursor
}

pub struct TensorIterator<'a, T>
where
    T: 'a
{
    tensor: &'a Tensor<T>,
    backend: IteratorBackend<'a>,
}