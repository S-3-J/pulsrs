pub mod bad_practices;
pub mod error_methods;
pub mod kernel_error;
pub mod mapping_err;
pub mod memory_err;
pub mod shape_err;
pub mod tensor_err;

use crate::dtype::DType;

use super::shape::Shape;

#[derive(Debug, PartialEq)]
pub enum PulsrsError {
    BadPractices(BadPractices),
    ShapeErr(ShapeError),
    TensorErr(TensorError),
    MemErr(MemoryError),
    //IterErr(IteratorError),
    //SlicingErr(SlicingError),
    MappingErr(MappingError),
    KernelErr(KernelError),
}

#[derive(Debug, PartialEq)]
pub enum BadPractices {
    UniqueDataConstraintError {
        given: Vec<usize>,
    },

    AxisOutofBounds {
        max: usize,
        min: usize,
        found: usize,
    },

    IndexOutofBounds {
        max: usize,
        found: usize,
    },
}

#[derive(Debug, PartialEq)]
pub enum ShapeError {
    RankMismatch { expected: usize, found: usize },

    ShapeIncompatiblewithElements { expected: usize, permitted: usize },
}

#[derive(Debug, PartialEq)]
pub enum TensorError {
    ReshapedElementMismatch { expected: usize, found: usize },

    TensorIndexingError { shape: Shape, indices: Vec<usize> },

    UncompatibleBroadcast { from: Shape, to: Shape },
}

#[derive(Debug, PartialEq)]
pub enum MappingError {
    BinaryMappingError {
        shape1: Shape,
        shape2: Shape,
        message: String,
    },
}

#[derive(Debug, PartialEq)]
pub enum IteratorError {}

#[derive(Debug, PartialEq)]
pub enum SlicingError {}

#[derive(Debug, PartialEq)]
pub enum MemoryError {
    NonContiguousMemoryError { function_path: &'static str },
}

#[derive(Debug, PartialEq)]
pub enum KernelError {
    SIMDIncompatibleDType { dtype: DType },

    SIMDOpUncompabtibleDType { dtype: DType, operation: String },
}
