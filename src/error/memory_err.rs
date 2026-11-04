use crate::error::PulsrsError;

use super::MemoryError;

use std::{error::Error, fmt::Display};

impl Error for MemoryError {}

impl Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
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

impl From<MemoryError> for PulsrsError {
    fn from(value: MemoryError) -> Self {
        Self::MemErr(value)
    }
}
