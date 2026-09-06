use std::{ops::Bound};

use crate::error::PulsrsError;

pub mod conversions;

#[derive(Clone, Copy, Debug)]
pub struct SliceRange {
    start: Bound<usize>,
    end: Bound<usize>,
    step: isize,
    was_none: bool,
}

impl SliceRange {
    pub fn resolve(self, lower_bound: usize, upper_bound: usize) -> Result<(usize, usize, isize, bool), PulsrsError> {
        // Calculate the actual start index (inclusive)
        let start_idx = match self.start {
            Bound::Unbounded => lower_bound,
            Bound::Included(s) => s,
            Bound::Excluded(s) => s + 1,
        };

        // Check if start index is beyond the valid range [0, upper_bound-1]
        if start_idx >= upper_bound {
            return Err(PulsrsError::IndexOutofBounds {
                max: upper_bound - 1,
                found: start_idx,
            });
        }

        // Convert SliceRange bounds to actual end index (inclusive)
        let end_index_inclusive = match self.end {
            Bound::Unbounded => upper_bound - 1,
            Bound::Included(s) => s,
            Bound::Excluded(s) => if s == 0 { 0 } else { s - 1 },
        };

        // Clamp end index to valid range [0, upper_bound-1]
        let end_clamped = end_index_inclusive.clamp(0, upper_bound - 1);

        // Ensure start <= end for valid range
        let start_clamped = start_idx.clamp(0, upper_bound - 1);
        if start_clamped > end_clamped {
            // Empty range - represent as a single element at the clamped position to avoid underflow
            let clamped_idx = start_clamped;
            Ok((clamped_idx, clamped_idx, 1, self.was_none))
        } else {
            Ok((start_clamped, end_clamped, self.step, self.was_none))
        }
    }
}