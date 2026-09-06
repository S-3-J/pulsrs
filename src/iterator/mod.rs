
use crate::{dtype::Element, iterator::iterator_backend::IteratorBackend};
use crate::{tensor::Tensor, cursor::IndexCursor};

pub mod indexiterator;
pub mod tensoriterator;
pub mod iteration_style;
pub mod iterator_backend;

pub struct IndexIterator {
    cursor: IndexCursor
}

pub struct TensorIterator<'a, T>
where
    T: 'a + Element
{
    tensor: &'a Tensor<T>,
    backend: IteratorBackend<'a>,
}