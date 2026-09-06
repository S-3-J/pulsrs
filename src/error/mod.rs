
pub mod error_methods;
use super::shape::Shape;

#[derive(Debug, PartialEq)]
pub enum PulsrsError{
    // universal errors
    RankMismatch {
        expected: usize,
        found: usize,
    },

    ReshapedElementMismatch {
        expected: usize,
        found: usize,
    },

    ShapeIncompatiblewithElements {
        expected: usize,
        permitted: usize,
    },

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

    //tensor errors
    TensorIndexingError {
        shape: Shape,
        indices: Vec<usize>,
    },

    NonContiguousMemoryError {
        function_path: &'static str,
    },

    // mapping errors
    BinaryMappingError {
        shape1: Shape,
        shape2: Shape,
        message: String,
    },
}