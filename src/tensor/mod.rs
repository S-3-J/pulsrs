#[allow(unused_imports)]
use crate::{
    buffer::Buffer,
    dtype::{DType, Element},
    error::PulsrsError,
    shape::Shape,
    stride::Stride,
};
use std::rc::Rc;

#[doc(hidden)]
pub mod binary_operations;

#[doc(hidden)]
pub mod constructors;

#[doc(hidden)]
pub mod contiguous;

#[doc(hidden)]
pub mod display;

#[doc(hidden)]
pub mod indexing;

#[doc(hidden)]
pub mod iterators;

#[doc(hidden)]
pub mod mappings;

#[doc(hidden)]
pub mod metadata;

#[doc(hidden)]
pub mod permutation;

#[doc(hidden)]
pub mod reduce_operations;

#[doc(hidden)]
pub mod reshape;

#[doc(hidden)]
pub mod unary_operations;

#[doc(hidden)]
pub mod slicing;

#[doc(hidden)]
pub mod broadcast;

#[doc(hidden)]
pub mod functional_operations;

#[derive(Clone, Debug)]
pub struct Tensor<T: Element> {
    buffer: Rc<Buffer<T>>,
    shape: Shape,
    stride: Stride,
    offset: usize,
    contiguous: bool,
    dtype: DType,
}
