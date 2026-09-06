use super::{DType, Element};

macro_rules! element_trait_impl {
    ($($variant:ident => ($t:ty, $name:expr)),* $(,)?) => {

        impl DType {
            pub fn name(&self) -> &'static str {
                match self {
                    $( DType::$variant => { $name }, )*
                }
            }
        }

        $(
            element_trait_impl!(@impl_element $variant => ($t));
        )*
    };

    (@impl_element Bool => ($t:ty)) => {
        impl Element for $t {
            #[inline(always)]
            fn dtype() -> DType {
                DType::Bool
            }

            fn zero() -> Self {
                false
            }

            fn one() -> Self {
                true
            }
        }
    };

    (@impl_element $variant:ident => ($t:ty)) => {
        impl Element for $t {
            #[inline(always)]
            fn dtype() -> DType {
                DType::$variant
            }

            fn zero() -> Self {
                0 as $t
            }

            fn one() -> Self {
                1 as $t
            }
        }
    };
}

element_trait_impl! {
    Float32 => (f32, "float32 / float"),
    Float64 => (f64, "float64 / double"),

    Int8    => ( i8,  "int8"),
    Int16   => ( i16, "int16"),
    Int32   => ( i32, "int32"),
    Int64   => ( i64, "int64 / long"),
    Int128  => (i128, "int128 / longlong"),

    Uint8   => ( u8,  "uint8"),
    Uint16  => ( u16, "uint16"),
    Uint32  => ( u32, "uint32"),
    Uint64  => ( u64, "uint64"),
    Uint128 => (u128, "uint128"),
    Bool    => (bool, "boolean")
}
