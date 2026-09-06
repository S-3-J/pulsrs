pub mod element_implementations;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DType {
    //float
    //Float16,
    Float32,
    Float64,

    //integers
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,

    //unsigned integer
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Uint128,

    //boolean
    Bool
}

#[allow(non_upper_case_globals)]
impl DType {
    // pytorch style constants
    // float
    //pub const Half: DType = Self::Float16;
    pub const Float: DType = Self::Float32;
    pub const Double: DType = Self::Float64;

    // integers
    pub const Long: DType = Self::Int64;
    pub const LongLong: DType = Self::Int128;
}

pub trait Element: Sized + Copy + 'static {
    fn dtype() -> DType;

    fn zero() -> Self;

    fn one() -> Self;
}
