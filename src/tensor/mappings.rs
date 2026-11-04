use std::{collections::HashSet, fmt::Debug, iter::zip};

use crate::{
    dtype::Element,
    error::{BadPractices, MappingError, PulsrsError},
    iterator::iteration_style::IterStyle,
    kernels::{BinarySimdKernel, FunctionSimdKernel, UnarySimdKernel},
    shape::Shape,
    simd::Vectorizable,
};

use super::Tensor;

impl<T: Vectorizable + Default> Tensor<T> {
    // a function applyer
    pub fn function_map<F, K>(&self, ops: F) -> Tensor<K>
    where
        T: Copy + Debug,
        F: Fn(T) -> K,
        K: Vectorizable,
    {
        let buffer: Vec<K> = if self.is_contiguous() {
            let mut buffer = vec![K::zero(); self.numel()];
            let fx_kernel = FunctionSimdKernel::<T, K>::try_new();

            fx_kernel.operate(self.buffer_as_slice().unwrap(), buffer.as_mut_slice(), ops);

            buffer
        } else {
            let mut buffer = Vec::<K>::with_capacity(self.numel());
            let mut elements = self.tensor_iterator(IterStyle::Cstyle);

            for e in &mut elements {
                buffer.push(ops(e))
            }

            buffer
        };

        Tensor::from_vector(buffer, self.shape().dims().to_vec()).unwrap()
    }

    // a binary function with a constant like, f(a, b) for all a that belong to tensor T and b is a constant.
    pub fn unary_map<F, K>(&self, other: T, ops: F) -> Tensor<K>
    where
        T: Copy + Debug,
        F: Fn(T, T) -> K,
        K: Vectorizable,
    {
        let buffer: Vec<K> = if self.contiguous {
            let mut buffer = vec![K::zero(); self.numel()];

            let unary_kernel = UnarySimdKernel::<T, K>::try_new();

            unary_kernel.operate(
                self.buffer_as_slice().unwrap(),
                other,
                buffer.as_mut_slice(),
                ops,
            );
            buffer
        } else {
            let mut buffer = Vec::with_capacity(self.numel());
            let mut elements = self.tensor_iterator(IterStyle::Cstyle);

            for e in &mut elements {
                buffer.push(ops(e, other))
            }

            buffer
        };

        Tensor::from_vector(buffer, self.shape().dims().to_vec()).unwrap()
    }

    // maps a binary operation, where operands are two corresponding elements from different tensors.
    pub fn elementwise_map<F, K>(&self, other: &Self, ops: F) -> Tensor<K>
    where
        T: Copy + Debug,
        F: Fn(T, T) -> K,
        K: Element,
    {
        if self.shape() != other.shape() {
            panic!("Tensor operation error: Incompatible shapes!.")
        }

        let mut buffer: Vec<K> = Vec::with_capacity(self.numel());
        let mut self_iterator = self.tensor_iterator(IterStyle::Cstyle);
        let mut other_iterator = other.tensor_iterator(IterStyle::Cstyle);

        for (a, b) in zip(&mut self_iterator, &mut other_iterator) {
            buffer.push(ops(a, b));
        }

        Tensor::from_vector(buffer, self.shape().dims().to_vec()).unwrap()
    }

    pub fn try_elementwise_map<F, K>(&self, other: &Self, ops: F) -> Result<Tensor<K>, PulsrsError>
    where
        T: Copy + Debug + Vectorizable,
        F: Fn(T, T) -> K,
        K: Element + Vectorizable,
    {
        if self.shape() != other.shape() {
            return Err(MappingError::BinaryMappingError {
                shape1: self.shape().clone(),
                shape2: other.shape().clone(),
                message: "Incompatible shapes.".to_string(),
            }
            .into());
        }

        let buffer: Vec<K> = if self.contiguous && other.contiguous {
            let binary_kernel = BinarySimdKernel::<T, K>::try_new();
            let mut buffer = vec![K::zero(); self.numel()];

            binary_kernel.operate(
                self.buffer_as_slice().unwrap(),
                other.buffer.as_slice(),
                buffer.as_mut_slice(),
                &ops,
            );
            buffer
        } else {
            let mut buffer = Vec::with_capacity(self.numel());
            let mut self_iterator = self.tensor_iterator(IterStyle::Cstyle);
            let mut other_iterator = other.tensor_iterator(IterStyle::Cstyle);

            for (a, b) in zip(&mut self_iterator, &mut other_iterator) {
                buffer.push(ops(a, b));
            }

            buffer
        };

        Tensor::from_vector(buffer, self.shape().dims().to_vec())
    }

    pub fn reduce<F>(&self, axes: &[usize], ops: F) -> Self
    where
        T: Copy + Debug,
        F: Fn(T, T) -> T,
    {
        if self.is_scalar() {
            self.clone()
        } else {
            let mut axes = axes.to_vec();

            if !axes.iter().all(|axis| *axis < self.ndim()) {
                // error here
                panic!("Axis out of bounds.")
            }

            let mut member = HashSet::new();

            if !axes.iter().all(|a| member.insert(*a)) {
                // error here
                panic!("Duplicate axis provided.")
            }

            axes.sort_unstable();

            let mut new_dims = self.shape().dims().to_vec();
            let reduction_width = axes.iter().map(|a| new_dims[*a]).product::<usize>();

            for axis in axes.iter().rev() {
                let _ = new_dims.remove(*axis);
            }

            let new_shape = Shape::new(new_dims);

            let mut buffer: Vec<T> = Vec::with_capacity(new_shape.numel());

            let mut reduction_priority: Vec<usize> = Vec::with_capacity(self.ndim());

            for axis in &axes {
                reduction_priority.push(*axis);
            }

            for i in (0..self.ndim()).rev() {
                if member.contains(&i) {
                    continue;
                }
                reduction_priority.push(i);
            }

            let mut offsetter = self.tensor_iterator(IterStyle::Custom(reduction_priority));

            for _i in 0..new_shape.numel() {
                let Some(mut acc) = offsetter.next() else {
                    break;
                };

                for _j in 1..reduction_width {
                    let Some(e) = offsetter.next() else {
                        std::panic!(
                            "TensorIterator returned None. This is not intended behaviour.\nReturned None at element: {}, step: {}. Total steps required: {}",
                            _i,
                            _j,
                            reduction_width
                        )
                    };

                    acc = ops(acc, e);
                }

                buffer.push(acc);
            }

            Tensor::from_vector(buffer, new_shape.dims().to_vec()).unwrap()
        }
    }

    pub fn try_reduce<F>(&self, axes: &[usize], ops: F) -> Result<Self, PulsrsError>
    where
        T: Copy + Debug,
        F: Fn(T, T) -> T,
    {
        if self.is_scalar() {
            Ok(self.clone())
        } else {
            let mut axes = axes.to_vec();

            for &axis in axes.iter() {
                if axis >= self.ndim() {
                    return Err(BadPractices::AxisOutofBounds {
                        max: self.ndim(),
                        min: 0,
                        found: axis,
                    }
                    .into());
                }
            }

            let mut member = HashSet::new();

            if !axes.iter().all(|a| member.insert(*a)) {
                return Err(BadPractices::UniqueDataConstraintError { given: axes }.into());
            }

            axes.sort_unstable();

            let mut new_dims = self.shape().dims().to_vec();
            let reduction_width = axes.iter().map(|a| new_dims[*a]).product::<usize>();

            for axis in axes.iter().rev() {
                let _ = new_dims.remove(*axis);
            }

            let new_shape = Shape::new(new_dims);

            let mut buffer: Vec<T> = Vec::with_capacity(new_shape.numel());

            let mut reduction_priority: Vec<usize> = Vec::with_capacity(self.ndim());

            for axis in &axes {
                reduction_priority.push(*axis);
            }

            for i in (0..self.ndim()).rev() {
                if member.contains(&i) {
                    continue;
                }
                reduction_priority.push(i);
            }

            let mut offsetter = self.tensor_iterator(IterStyle::Custom(reduction_priority));

            for _i in 0..new_shape.numel() {
                let Some(mut acc) = offsetter.next() else {
                    break;
                };

                for _j in 1..reduction_width {
                    let Some(e) = offsetter.next() else {
                        std::panic!(
                            "TensorIterator returned None. This is not intended behaviour.\nReturned None at element: {}, step: {}. Total steps required: {}",
                            _i,
                            _j,
                            reduction_width
                        )
                    };

                    acc = ops(acc, e);
                }

                buffer.push(acc);
            }

            Tensor::from_vector(buffer, new_shape.dims().to_vec())
        }
    }
}
