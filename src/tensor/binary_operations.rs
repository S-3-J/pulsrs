
use std::{
    fmt::Debug, ops::{Add, Div, Mul, Sub},
};

use crate::{dtype::Element, error::PulsrsError, tensor::Tensor};

impl<T> Tensor<T> 
where
    T: Element + Debug + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{
    pub fn try_add(&self, other: &Self) -> Result<Self, PulsrsError> {
        self.try_elementwise_map(other, |a, b| a + b)
    }

    pub fn try_sub(&self, other: &Self) -> Result<Self, PulsrsError> {
        self.try_elementwise_map(other, |a, b| a - b)
    }

    pub fn try_mul(&self, other: &Self) -> Result<Self, PulsrsError> {
        self.try_elementwise_map(other, |a, b| a * b)
    }

    pub fn try_div(&self, other: &Self) -> Result<Self, PulsrsError> {
        self.try_elementwise_map(other, |a, b| a / b)
    }
}


impl<'a, 'b, T> Add<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Add<Output = T> + Debug + Element
{
    type Output = Tensor<T>;

    fn add(self, rhs: &'b Tensor<T>) -> Self::Output {
        self.elementwise_map(rhs, |a, b| a + b)
    }
}

impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Sub<Output = T> + Debug + Element
{
    type Output = Tensor<T>;

    fn sub(self, rhs: &'b Tensor<T>) -> Self::Output {
        self.elementwise_map(rhs, |a, b| a - b)
    }
}

impl<'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Mul<Output = T> + Debug + Element
{
    type Output = Tensor<T>;

    fn mul(self, rhs: &'b Tensor<T>) -> Self::Output {
        self.elementwise_map(rhs, |a, b| a * b)
    }
}

impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Div<Output = T> + Debug + Element
{
    type Output = Tensor<T>;

    fn div(self, rhs: &'b Tensor<T>) -> Self::Output {
        self.elementwise_map(rhs, |a, b| a / b)
    }
}

impl<T: Element> Tensor<T> {
    pub fn eq(&self, other: &Self) -> Tensor<bool> 
    where 
        T: PartialEq + Debug
    {
        self.elementwise_map(other, |a, b| a == b)
    }
}
