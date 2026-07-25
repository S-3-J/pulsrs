
#[allow(unused_imports)]
use crate::{tensor::Tensor, error::PulsrsError, cursor::IndexCursor};

pub mod indexiterator;
pub mod tensoriterator;
pub mod iteration_style;

pub struct IndexIterator {
    cursor: IndexCursor
}

pub struct TensorIterator<'a, T>
where
    T: 'a
{
    tensor: &'a Tensor<T>,
    cursor: IndexCursor,
}