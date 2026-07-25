
#[allow(unused_imports)]
use crate::{ shape::Shape, stride::Stride, buffer::Buffer, error::PulsrsError};
use std::rc::Rc;

pub mod constructors;
pub mod metadata;
pub mod contiguous;
pub mod reshape;
pub mod permutation;
pub mod indexing;
pub mod iterators;

#[derive(Clone, Debug)]
pub struct Tensor<T>
{
    buffer: Rc<Buffer<T>>,
    shape: Shape,
    stride: Stride,
    contiguous: bool,
}