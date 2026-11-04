use std::{error::Error, fmt::Display};

use crate::error::PulsrsError;

use super::KernelError;

impl Error for KernelError {}

impl Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use KernelError::*;
        match self {
            SIMDIncompatibleDType { dtype } => {
                let a = write!(
                    f,
                    "Selected primitive, {} doesn't support SIMD Operations.",
                    dtype.name()
                );

                a
            }
            SIMDOpUncompabtibleDType { dtype, operation } => {
                let a = write!(
                    f,
                    "Simd operation uncompatible data type: Selected primitve, {} doesn't support SIMD for {} operation.",
                    dtype.name(),
                    operation
                );

                a
            }
        }
    }
}

impl From<KernelError> for PulsrsError {
    fn from(value: KernelError) -> Self {
        Self::KernelErr(value)
    }
}
