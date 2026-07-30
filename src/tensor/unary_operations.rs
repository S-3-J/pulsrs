
use std::{
    ops::{Add, Div, Mul, Rem, Sub}
};
use crate::tensor::Tensor;

macro_rules! impl_scalar_operations {
    ($($t:ty),*) => {
        $(
            impl<'a> Add<$t> for &'a Tensor<$t>
            {
                type Output = Tensor<$t>;

                fn add(self, rhs: $t) -> Self::Output {
                    self.unary_map(rhs, |a, b| a + b)
                }
            }

            impl<'a> Sub<$t> for &'a Tensor<$t>
            {
                type Output = Tensor<$t>; 

                fn sub(self, rhs: $t) -> Self::Output {
                    self.unary_map(rhs, |a, b| a - b)
                }
            }

            impl<'a> Mul<$t> for &'a Tensor<$t>
            {
                type Output = Tensor<$t>;

                fn mul(self, rhs: $t) -> Self::Output {
                    self.unary_map(rhs, |a, b| a * b)
                }
            }

            impl<'a> Div<$t> for &'a Tensor<$t>
            {
                type Output = Tensor<$t>;

                fn div(self, rhs: $t) -> Self::Output {
                    self.unary_map(rhs, |a, b| a / b)
                }
            }

            impl<'a> Rem<$t> for &'a Tensor<$t>
            {
                type Output = Tensor<$t>;

                fn rem(self, rhs: $t) -> Self::Output {
                    self.unary_map(rhs, |a, b| a % b)
                }
            }       
        )*
    };
}

impl_scalar_operations!(f32, f64, i32, i64, u32, u64);