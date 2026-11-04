use super::PulsrsError;
use std::error::Error;
use std::fmt::{self};

impl Error for PulsrsError {}

impl fmt::Display for PulsrsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadPractices(e) => e.fmt(f),
            //Self::IterErr(e) => e.fmt(f),
            Self::MappingErr(e) => e.fmt(f),
            Self::MemErr(e) => e.fmt(f),
            Self::ShapeErr(e) => e.fmt(f),
            //Self::SlicingErr(e) => e.fmt(f),
            Self::TensorErr(e) => e.fmt(f),
            Self::KernelErr(e) => e.fmt(f),
        }
    }
}
