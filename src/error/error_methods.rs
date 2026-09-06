
use std::fmt;
use std::error::Error;
use super::PulsrsError;

impl Error for PulsrsError { }

impl fmt::Display for PulsrsError {
    fn fmt(
        &self, 
        f: &mut std::fmt::Formatter<'_>
    ) -> std::fmt::Result {
        match self {
            //universal errors
            Self::RankMismatch { expected, found } => {
               write!(
                f,
                "Rank Mismatch : expected {} dimension, found {}",
                expected,
                found
               )?;
               Ok(())
            },

            Self::ReshapedElementMismatch { expected, found } => {
                write!(
                    f,
                    "Reshaped element mismatch: expected {} elements, found {}",
                    expected,
                    found
                )?;
                Ok(())
            },

            Self::ShapeIncompatiblewithElements { expected, permitted } => {
                write!(
                    f,
                    "Shape incomaptible with element count: expected {} elements, shape permits {}",
                    expected,
                    permitted
                )?;
                Ok(())
            },

            PulsrsError::UniqueDataConstraintError { given } => {
                write!(
                    f,
                    "Unique data constraint error: requires values to be unique, given {:?}",
                    given
                )?;
                Ok(())
            },

            Self::AxisOutofBounds { max, min, found } => {
                write!(
                    f,
                    "Axis out of bounds: requires axis values between {} and {}, found {}",
                    min,
                    max,
                    found
                )?;
                Ok(())
            },

            Self::IndexOutofBounds { max, found } => {
                write!(
                    f,
                    "Index out of bounds: Cannot index {}, where limit is {}",
                    found,
                    max
                )?;
                Ok(())
            },

            //tensor errors
            Self::TensorIndexingError { shape, indices } => {
                write!(
                    f,
                    "Invalid indexing: shape of the tensor is {}, received indices {:?}",
                    shape,
                    indices
                )?;
                Ok(())
            },

            Self::BinaryMappingError { shape1, shape2, message } => {
                write!(
                    f,
                    "Binary mapping error: Failed to perform elementwise map, on tensors of shape {} and {}. \nAdditonal Information: {}",
                    shape1,
                    shape2,
                    message
                )?;
                Ok(())
            },

            Self::NonContiguousMemoryError { function_path } => {
                write!(
                    f,
                    "Failed to execute this particular operation, it requires memory to be contiguous.\nError invoked by: {}",
                    function_path
                )?;
                Ok(())
            }
        }    
    }
}