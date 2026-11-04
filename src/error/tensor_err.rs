use std::{error::Error, fmt::Display};

use crate::error::PulsrsError;

use super::TensorError;

impl Error for TensorError {}

impl Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReshapedElementMismatch { expected, found } => {
                write!(
                    f,
                    "Reshaped element mismatch: expected {} elements, found {}",
                    expected, found
                )?;
                Ok(())
            }

            Self::TensorIndexingError { shape, indices } => {
                write!(
                    f,
                    "Invalid indexing: shape of the tensor is {}, received indices {:?}",
                    shape, indices
                )?;
                Ok(())
            }

            Self::UncompatibleBroadcast { from, to } => {
                write!(
                    f,
                    "Uncompatible Broadcast: cannot broadcast a tensor of shape {}, to {}.\nCheck for unequal dimensions that are not degenerate / singleton.",
                    from, to
                )?;
                Ok(())
            }
        }
    }
}

impl From<TensorError> for PulsrsError {
    fn from(value: TensorError) -> Self {
        Self::TensorErr(value)
    }
}
