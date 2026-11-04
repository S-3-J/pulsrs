use core::panic;
use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

use crate::{
    error::{PulsrsError, TensorError},
    shape::Shape,
    simd::Vectorizable,
    tensor::Tensor,
};

impl<T> Tensor<T>
where
    T: Vectorizable
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Default,
{
    pub fn try_add(&self, other: &Self) -> Result<Self, PulsrsError> {
        if self.shape() != other.shape() {
            let (safe, final_shape) = self.is_broadcastible_for(other.shape().dims());

            if safe {
                self.broadcast(&final_shape)
                    .try_elementwise_map(&other.broadcast(&final_shape), |a, b| a + b)
            } else {
                Err(TensorError::UncompatibleBroadcast {
                    from: self.shape().clone(),
                    to: Shape::new(final_shape),
                }
                .into())
            }
        } else {
            self.try_elementwise_map(other, |a, b| a + b)
        }
    }

    pub fn try_sub(&self, other: &Self) -> Result<Self, PulsrsError> {
        if self.shape() != other.shape() {
            let (safe, final_shape) = self.is_broadcastible_for(other.shape().dims());

            if safe {
                self.broadcast(&final_shape)
                    .try_elementwise_map(&other.broadcast(&final_shape), |a, b| a - b)
            } else {
                Err(TensorError::UncompatibleBroadcast {
                    from: self.shape().clone(),
                    to: Shape::new(final_shape),
                }
                .into())
            }
        } else {
            self.try_elementwise_map(other, |a, b| a - b)
        }
    }

    pub fn try_mul(&self, other: &Self) -> Result<Self, PulsrsError> {
        if self.shape() != other.shape() {
            let (safe, final_shape) = self.is_broadcastible_for(other.shape().dims());

            if safe {
                self.broadcast(&final_shape)
                    .try_elementwise_map(&other.broadcast(&final_shape), |a, b| a * b)
            } else {
                Err(TensorError::UncompatibleBroadcast {
                    from: self.shape().clone(),
                    to: Shape::new(final_shape),
                }
                .into())
            }
        } else {
            self.try_elementwise_map(other, |a, b| a * b)
        }
    }

    pub fn try_div(&self, other: &Self) -> Result<Self, PulsrsError> {
        if self.shape() != other.shape() {
            let (safe, final_shape) = self.is_broadcastible_for(other.shape().dims());

            if safe {
                self.broadcast(&final_shape)
                    .try_elementwise_map(&other.broadcast(&final_shape), |a, b| a / b)
            } else {
                Err(TensorError::UncompatibleBroadcast {
                    from: self.shape().clone(),
                    to: other.shape().clone(),
                }
                .into())
            }
        } else {
            self.try_elementwise_map(other, |a, b| a / b)
        }
    }
}

impl<'a, 'b, T> Add<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Add<Output = T> + Debug + Vectorizable + Default,
{
    type Output = Tensor<T>;

    fn add(self, rhs: &'b Tensor<T>) -> Self::Output {
        if self.shape() != rhs.shape() {
            let (safe, final_shape) = self.is_broadcastible_for(rhs.shape().dims());

            if safe {
                let r = self
                    .broadcast(&final_shape)
                    .try_elementwise_map(&rhs.broadcast(&final_shape), |a, b| a + b);

                if let Ok(ans) = r {
                    ans
                } else {
                    panic!("{}", r.unwrap_err())
                }
            } else {
                panic!(
                    "{}",
                    TensorError::UncompatibleBroadcast {
                        from: self.shape().clone(),
                        to: rhs.shape().clone()
                    }
                )
            }
        } else {
            let r = self.try_elementwise_map(rhs, |a, b| a + b);

            if let Ok(ans) = r {
                ans
            } else {
                panic!("{}", r.unwrap_err())
            }
        }
    }
}

impl<'a, 'b, T> Sub<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Sub<Output = T> + Debug + Vectorizable + Default,
{
    type Output = Tensor<T>;

    fn sub(self, rhs: &'b Tensor<T>) -> Self::Output {
        if self.shape() != rhs.shape() {
            let (safe, final_shape) = self.is_broadcastible_for(rhs.shape().dims());

            if safe {
                let r = self
                    .broadcast(&final_shape)
                    .try_elementwise_map(&rhs.broadcast(&final_shape), |a, b| a - b);

                if let Ok(ans) = r {
                    ans
                } else {
                    panic!("{}", r.unwrap_err())
                }
            } else {
                panic!(
                    "{}",
                    TensorError::UncompatibleBroadcast {
                        from: self.shape().clone(),
                        to: rhs.shape().clone()
                    }
                )
            }
        } else {
            let r = self.try_elementwise_map(rhs, |a, b| a - b);

            if let Ok(ans) = r {
                ans
            } else {
                panic!("{}", r.unwrap_err())
            }
        }
    }
}

impl<'a, 'b, T> Mul<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Mul<Output = T> + Debug + Vectorizable + Default,
{
    type Output = Tensor<T>;

    fn mul(self, rhs: &'b Tensor<T>) -> Self::Output {
        if self.shape() != rhs.shape() {
            let (safe, final_shape) = self.is_broadcastible_for(rhs.shape().dims());

            if safe {
                let r = self
                    .broadcast(&final_shape)
                    .try_elementwise_map(&rhs.broadcast(&final_shape), |a, b| a * b);

                if let Ok(ans) = r {
                    ans
                } else {
                    panic!("{}", r.unwrap_err())
                }
            } else {
                panic!(
                    "{}",
                    TensorError::UncompatibleBroadcast {
                        from: self.shape().clone(),
                        to: rhs.shape().clone()
                    }
                )
            }
        } else {
            let r = self.try_elementwise_map(rhs, |a, b| a * b);

            if let Ok(ans) = r {
                ans
            } else {
                panic!("{}", r.unwrap_err())
            }
        }
    }
}

impl<'a, 'b, T> Div<&'b Tensor<T>> for &'a Tensor<T>
where
    T: 'a + 'b + Copy + Div<Output = T> + Debug + Vectorizable + Default,
{
    type Output = Tensor<T>;

    fn div(self, rhs: &'b Tensor<T>) -> Self::Output {
        if self.shape() != rhs.shape() {
            let (safe, final_shape) = self.is_broadcastible_for(rhs.shape().dims());

            if safe {
                let r = self
                    .broadcast(&final_shape)
                    .try_elementwise_map(&rhs.broadcast(&final_shape), |a, b| a / b);

                if let Ok(ans) = r {
                    ans
                } else {
                    panic!("{}", r.unwrap_err())
                }
            } else {
                panic!(
                    "{}",
                    TensorError::UncompatibleBroadcast {
                        from: self.shape().clone(),
                        to: rhs.shape().clone()
                    }
                )
            }
        } else {
            let r = self.try_elementwise_map(rhs, |a, b| a / b);

            if let Ok(ans) = r {
                ans
            } else {
                panic!("{}", r.unwrap_err())
            }
        }
    }
}

impl<T: Vectorizable> Tensor<T> {
    pub fn eq(&self, other: &Self) -> Tensor<bool>
    where
        T: PartialEq + Debug + Default,
    {
        let r = self.try_elementwise_map(other, |a, b| a == b);

        if let Ok(ans) = r {
            ans
        } else {
            panic!("{}", r.unwrap_err())
        }
    }
}
