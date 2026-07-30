
use std::{fmt::Debug, ops::{Add, Div, Mul}};

use super::Tensor;
use num_traits::FromPrimitive;


// simple methods
impl<T> Tensor<T> 
where 
    T: Copy + Debug
{
    pub fn reduce_sum(&self, axes: &[usize]) -> Self 
    where 
        T: Add<Output = T>
    {
        self.reduce(axes, |a, b| a + b)
    }

    pub fn reduce_prod(&self, axes: &[usize]) -> Self
    where 
        T: Mul<Output = T>
    {
        self.reduce(axes, |a, b| a * b)
    }

    pub fn reduce_max(&self, axes: &[usize]) -> Self
    where
        T: PartialOrd
    {
        self.reduce(axes, |a, b| {
            if a >= b {
                a
            } else {
                b
            }
        })
    }

    pub fn reduce_min(&self, axes: &[usize]) -> Self
    where
        T: PartialOrd
    {
        self.reduce(axes, |a, b| {
            if a >= b {
                b
            } else {
                a
            }
        })
    }
}

//complex methods
impl<T> Tensor<T> 
where
    T: Copy + Debug
{
    pub fn reduce_mean(&self, axes: &[usize]) -> Self
    where 
        T: Add<Output = T> + FromPrimitive,
        for<'a> &'a Tensor<T>: Div<T, Output = Tensor<T>>
    {   
        let dims = self.shape().dims().to_vec();
        let d = axes.iter().map(|&a| dims[a]).product::<usize>(); 
        let rs = self.reduce_sum(axes);

        let d_scalar = T::from_usize(d).expect("An error occured parsing while parsing usize to scalar in reduce mean.");
        
        &rs / d_scalar
    }
}