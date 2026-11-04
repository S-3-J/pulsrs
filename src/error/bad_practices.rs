use crate::error::PulsrsError;
use std::{error::Error, fmt::Display};

use super::BadPractices;

impl Error for BadPractices {}

impl Display for BadPractices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AxisOutofBounds { max, min, found } => {
                write!(
                    f,
                    "Axis out of bounds: requires axis values between {} and {}, found {}",
                    min, max, found
                )?;
                Ok(())
            }

            Self::IndexOutofBounds { max, found } => {
                write!(
                    f,
                    "Index out of bounds: Cannot index {}, where limit is {}",
                    found, max
                )?;
                Ok(())
            }
            Self::UniqueDataConstraintError { given } => {
                write!(
                    f,
                    "Unique data constraint error: requires values to be unique, given {:?}",
                    given
                )?;
                Ok(())
            }
        }
    }
}

impl From<BadPractices> for PulsrsError {
    fn from(value: BadPractices) -> Self {
        Self::BadPractices(value)
    }
}
