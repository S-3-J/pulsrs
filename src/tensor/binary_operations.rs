
use std::{
    fmt::Debug, ops::{Add, Div, Mul, Sub},
};

use crate::tensor::Tensor;

impl<'a, 'b, T> Add<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Add<Output = T> + Debug
{
    type Output = Tensor<T>;

    fn add(self, rhs: &'b Tensor<T>) -> Self::Output {
        self.elementwise_map(rhs, |a, b| a + b)
    }
}

impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Sub<Output = T> + Debug
{
    type Output = Tensor<T>;

    fn sub(self, rhs: &'b Tensor<T>) -> Self::Output {
        self.elementwise_map(rhs, |a, b| a - b)
    }
}

impl<'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Mul<Output = T> + Debug
{
    type Output = Tensor<T>;

    fn mul(self, rhs: &'b Tensor<T>) -> Self::Output {
        self.elementwise_map(rhs, |a, b| a * b)
    }
}

impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Div<Output = T> + Debug
{
    type Output = Tensor<T>;

    fn div(self, rhs: &'b Tensor<T>) -> Self::Output {
        self.elementwise_map(rhs, |a, b| a / b)
    }
}