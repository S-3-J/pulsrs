#[allow(unused_imports)]
use crate::{
    buffer::Buffer,
    dtype::{DType, Element},
    error::PulsrsError,
    shape::Shape,
    stride::Stride,
};
use std::rc::Rc;

pub mod binary_operations;
pub mod broadcast;
pub mod constructors;
pub mod contiguous;
pub mod display;
pub mod indexing;
pub mod iterators;
pub mod mappings;
pub mod metadata;
pub mod permutation;
pub mod reduce_operations;
pub mod reshape;
pub mod slicing;
pub mod unary_operations;

#[derive(Clone, Debug)]
pub struct Tensor<T: Element> {
    buffer: Rc<Buffer<T>>,
    shape: Shape,
    stride: Stride,
    offset: usize,
    contiguous: bool,
    dtype: DType,
}
