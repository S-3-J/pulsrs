use crate::error::PulsrsError;

use super::MappingError;

use std::{error::Error, fmt::Display};

impl Error for MappingError {}

impl Display for MappingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BinaryMappingError {
                shape1,
                shape2,
                message,
            } => {
                write!(
                    f,
                    "Binary mapping error: Failed to perform elementwise map, on tensors of shape {} and {}. \nAdditonal Information: {}",
                    shape1, shape2, message
                )?;
                Ok(())
            }
        }
    }
}

impl From<MappingError> for PulsrsError {
    fn from(value: MappingError) -> Self {
        Self::MappingErr(value)
    }
}
