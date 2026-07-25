pub mod stride;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stride {
    strides : Vec<usize>,
}