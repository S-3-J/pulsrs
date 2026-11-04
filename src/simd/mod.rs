use crate::dtype::Element;

pub(crate) trait Sealed {}

#[allow(private_bounds)]
pub trait Vectorizable: Element + Sealed {}

macro_rules! impl_vectz {
    ($($t:ty),*) => {
        $(
            impl Sealed for $t {}

            impl Vectorizable for $t { }
        )*
    };
}

impl_vectz!(
    u8, u16, u32, u64, f32, f64, i8, i16, i32, i64, u128, i128, bool
);
