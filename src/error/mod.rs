
pub mod error;
use super::shape::Shape;

#[derive(Debug, PartialEq)]
pub enum PulsrsError{
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

    TensorIndexingError {
        shape: Shape,
        indices: Vec<usize>,
    },

    UniqueDataConstraintError {
        given: Vec<usize>,
    },

    AxisOutofBounds {
        max: usize,
        min: usize,
        found: usize,
    }
}