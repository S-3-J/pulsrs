use crate::{cursor::IndexCursor, tensor::Tensor};
use crate::{dtype::Element, iterator::iterator_backend::IteratorBackend};

pub mod indexiterator;
pub mod iteration_style;
pub mod iterator_backend;
pub mod tensoriterator;

pub struct IndexIterator {
    cursor: IndexCursor,
}

pub struct TensorIterator<'a, T>
where
    T: 'a + Element,
{
    tensor: &'a Tensor<T>,
    backend: IteratorBackend<'a>,
}
