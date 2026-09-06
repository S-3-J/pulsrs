use super::SliceRange;
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive
};

pub trait IntoOptSliceRange {
    fn iosr(self) -> Option<SliceRange>;
}

macro_rules! impl_into_opt_slice_range {
    ($($t:ty),*) => {
        $(
            impl IntoOptSliceRange for $t {
                fn iosr(self) -> Option<SliceRange> {
                    Some(SliceRange {
                        start: self.start_bound().cloned(),
                        end: self.end_bound().cloned(),
                        step: 1,
                        was_none: false
                    })
                }
            }

            impl IntoOptSliceRange for ($t, isize) {
                fn iosr(self) -> Option<SliceRange> {
                    Some(SliceRange {
                        start: self.0.start_bound().cloned(),
                        end: self.0.end_bound().cloned(),
                        step: self.1,
                        was_none: false
                    })
                }
            }
        )*
    };
}

impl_into_opt_slice_range!(
    Range<usize>,
    RangeFrom<usize>,
    RangeFull,
    RangeInclusive<usize>,
    RangeTo<usize>,
    RangeToInclusive<usize>
);

impl IntoOptSliceRange for usize {
    fn iosr(self) -> Option<SliceRange> {
        Some(SliceRange {
            start: Bound::Included(self),
            end: Bound::Excluded(self + 1),
            step: 1,
            was_none: false
        })
    }
}