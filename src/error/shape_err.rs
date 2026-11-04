use std::{error::Error, fmt::Display};

use crate::error::PulsrsError;

use super::ShapeError;

impl Error for ShapeError {}

impl Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RankMismatch { expected, found } => {
                write!(
                    f,
                    "Rank Mismatch : expected {} dimension, found {}",
                    expected, found
                )?;
                Ok(())
            }

            Self::ShapeIncompatiblewithElements {
                expected,
                permitted,
            } => {
                write!(
                    f,
                    "Shape incomaptible with element count: expected {} elements, shape permits {}",
                    expected, permitted
                )?;
                Ok(())
            }
        }
    }
}

impl From<ShapeError> for PulsrsError {
    fn from(value: ShapeError) -> Self {
        Self::ShapeErr(value)
    }
}
