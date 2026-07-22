
use std::fmt;
use std::error::Error;
use crate::shape::shape::Shape;

#[derive(Debug, PartialEq)]
pub enum PulsrsError {
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

impl Error for PulsrsError { }

impl fmt::Display for PulsrsError {
    fn fmt(
        &self, 
        f: &mut std::fmt::Formatter<'_>
    ) -> std::fmt::Result {
        match self {
            PulsrsError::RankMismatch { expected, found } => {
               write!(
                f,
                "Rank Mismatch : expected {} dimension, found {}",
                expected,
                found
               ) 
            },
            PulsrsError::ReshapedElementMismatch { expected, found } => {
                write!(
                    f,
                    "Reshaped element mismatch: expected {} elements, found {}",
                    expected,
                    found
                )
            },
            PulsrsError::ShapeIncompatiblewithElements { expected, permitted } => {
                write!(
                    f,
                    "Shape incomaptible with element count: expected {} elements, shape permits {}",
                    expected,
                    permitted
                )
            },
            PulsrsError::TensorIndexingError { shape, indices } => {
                write!(
                    f,
                    "Invalid indexing: shape of the tensor is {}, received indices {:?}",
                    shape,
                    indices
                )
            },
            PulsrsError::UniqueDataConstraintError { given } => {
                write!(
                    f,
                    "Unique data constraint error: requires values to be unique, given {:?}",
                    given
                )
            },
            PulsrsError::AxisOutofBounds { max, min, found } => {
                write!{
                    f,
                    "Axis out of bounds: requires axis values between {} and {}, found {}",
                    min,
                    max,
                    found
                }
            }
        }    
    }
}