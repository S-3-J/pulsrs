pub mod stride_methods;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stride {
    strides : Vec<isize>,
}