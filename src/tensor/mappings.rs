use std::{collections::HashSet, fmt::Debug, iter::zip};

use crate::{
    iterator::iteration_style::IterStyle, 
    shape::Shape
};

use super::Tensor;

impl<T> Tensor<T> {

    pub fn unary_map<F>(
        &self,
        other: T,
        ops: F,
    ) -> Self
    where 
        T: Copy + Debug,
        F: Fn(T,T) -> T
    {
        let mut buffer: Vec<T> = Vec::with_capacity(self.numel());
        let mut elements = self.tensor_iterator(IterStyle::Cstyle);

        for e in &mut elements {
            buffer.push(ops(e, other))
        }

        Self::from_vector(buffer, self.shape().dims().to_vec()).unwrap()
    }

    pub fn elementwise_map<F>(
        &self,
        other: &Self,
        ops: F,
    ) -> Self
    where
        T: Copy + Debug,
        F: Fn(T, T) -> T
    {

        if self.shape() != other.shape() {
            panic!("Tensor operation error: Incompatible shapes!.")
        }

        let mut buffer: Vec<T> = Vec::with_capacity(self.numel());
        let mut self_iterator = self.tensor_iterator(IterStyle::Cstyle);
        let mut other_iterator = other.tensor_iterator(IterStyle::Cstyle);

        for (a, b) in zip(&mut self_iterator, &mut other_iterator) {
            buffer.push(ops(a, b));
        }

        Self::from_vector(buffer, self.shape().dims().to_vec()).unwrap()
    }

    pub fn reduce<F>(
        &self,
        axes: &[usize],
        ops: F,
    ) -> Self
    where 
        T: Copy + Debug,
        F: Fn(T, T) -> T
    {
        if self.is_scalar() {
            self.clone()
        } else {
            let mut axes = axes.to_vec();

            if !axes.iter().all(|axis| *axis < self.ndim()) {
                panic!("Axis out of bounds.")
            }

            let mut member = HashSet::new();

            if !axes.iter().all(|a| member.insert(*a)) {
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
                    break
                };

                for _j in 1..reduction_width {
                    let Some(e) = offsetter.next() else {
                        panic!("This shouldn't happen")
                    };  

                    acc = ops(
                        acc,
                        e
                    )
                }

                buffer.push(acc);
            }

            Tensor::from_vector(buffer, new_shape.dims().to_vec()).unwrap()
        }
    }
}


