
use crate::{ shape::shape::Shape, stride::stride::Stride, buffer::buffer::Buffer, error::error::PulsrsError};
use std::{collections::HashSet, rc::Rc};

#[derive(Clone)]
pub struct Tensor<T>
{
    buffer: Rc<Buffer<T>>,
    shape: Shape,
    stride: Stride,
    contiguous: bool,
}

impl<T> Tensor<T> 
where
    T: Copy + Default
{
    // give vector and shape
    pub fn from_vector(data: Vec<T>, dims: Vec<usize>) -> Result<Self, PulsrsError> {
        let shape =Shape::new(dims);
        
        if shape.numel() != data.len() {
            return Err(PulsrsError::ShapeIncompatiblewithElements { expected: data.len(), permitted: shape.numel() });
        }

        let buffer = Buffer::from_vec(data);
        let stride = Stride::from_shape(&shape);
        
        Ok(
            Tensor {
                buffer: Rc::new(buffer),
                shape,
                stride,
                contiguous: true,
            }
        )
    }

    pub fn new_scalar(data: T) -> Self {
        Tensor { buffer: Rc::new(Buffer::from_vec(vec![data])), shape: Shape::default(), stride: Stride::from(vec![]), contiguous: true}
    }

    // calculate rank
    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn ndim(&self) -> usize {
        self.rank()
    }

    // shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    // strides
    pub fn strides(&self) -> &[usize] {
        self.stride.strides()
    }

    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    // scalar check
    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn contiguous(&self) -> Self {

        if self.is_scalar() {
            return self.clone();
        }

        if self.is_contiguous() {
            return self.clone();
        } else {
            let mut contiguous_buffer: Vec<T> = vec![T::default(); self.numel()];
            let dims = self.shape().dims();
            
            let new_stride = Stride::from_shape(&self.shape);

            let mut index: Vec<usize> = vec![0; dims.len()];
            
            fn n_dim_iterator<T>(t: &Tensor<T>, dims: &[usize], depth: usize, index: &mut Vec<usize>, new_buffer: &mut Vec<T>, new_stride: &Stride)
            where
                T: Copy + Default
            {
                if depth < dims.len() - 1 {
                    n_dim_iterator(t, dims,depth+1, index, new_buffer, new_stride);
                }

                for i in 0..dims[depth] {

                    if depth+1 == dims.len() {
                        index[depth] = i;
                        
                        // should panic and crash if erroneous; 
                        //however its not possible to crash here as the tensor won't index out of bounds.
                        let val = t.get(index).unwrap();
                        let new_offset = new_stride.offset(&index).unwrap();
                        
                        new_buffer[new_offset] = val;   
                    } else {
                        index[depth] = i;
                        n_dim_iterator(t, dims, depth+1, index, new_buffer, new_stride);    
                    }

                }
            }
            
            n_dim_iterator(
                self, 
                dims, 
                0, 
                &mut index, 
                &mut contiguous_buffer, 
                &new_stride
            );

            
            Tensor {
                buffer: Rc::new(Buffer::from_vec(contiguous_buffer)),
                shape: self.shape.clone(), 
                stride: new_stride, 
                contiguous: true,
            }
        
        }
    }

    //reshape, have to implement via contiguous.
    pub fn reshape(&self, dims: Vec<usize>) -> Result<Self, PulsrsError> {
        
        let new_shape: Shape = dims.into();

        if self.shape.numel() != new_shape.numel() {
            
            Err(
                PulsrsError::ReshapedElementMismatch { 
                    expected: self.shape.numel(), 
                    found: new_shape.numel() 
                }
            )
        
        } else {

            let new_strides = Stride::from_shape(&new_shape);
            let mut t = self.contiguous();

            t.shape = new_shape;
            t.stride = new_strides;

            Ok(
                t
            )

        }

    }

    //permute (also transpose)
    pub fn permute(&self, permutation: Vec<usize>) -> Result<Self, PulsrsError> {

        if self.is_scalar() {
            if permutation.len() > 0 {
                return Err(
                    PulsrsError::RankMismatch { expected: 0, found: permutation.len() }
                );
            }
            return Ok(self.clone());
        }

        let mut index_set: HashSet<usize> = HashSet::new();

        for i in &permutation {
            if i >= &self.rank() {
                return Err(
                    PulsrsError::AxisOutofBounds { max: self.rank(), min: 0, found: *i }
                )
            }
            if !index_set.insert(*i) {
                return Err(
                    PulsrsError::UniqueDataConstraintError { given: permutation }
                );
            }
        }

        let curr_shape_dim = self.shape().dims();
        let curr_strides = self.strides();

        if curr_shape_dim.len() != permutation.len() {
            Err(
                PulsrsError::RankMismatch { 
                    expected: curr_shape_dim.len(), 
                    found: permutation.len(),
                }
            )
        } else {

            let mut permuted_shape = vec![0; curr_shape_dim.len()];
            let mut permuted_stride = vec![1; curr_strides.len()];

            for i in 0..curr_shape_dim.len() {
                permuted_shape[i] = curr_shape_dim[permutation[i]];
                permuted_stride[i] = curr_strides[permutation[i]];
            }
            
            let permuted_shape: Shape = permuted_shape.into(); 
            let permuted_stride: Stride = permuted_stride.into();

            let contiguous_flag = permuted_shape.dims() == self.shape().dims();

            Ok(
                Tensor {
                    buffer: Rc::clone(&self.buffer),
                    shape: permuted_shape,
                    stride: permuted_stride,
                    contiguous: contiguous_flag,
                }
            )
        }

    }

    pub fn get(&self, indices: &[usize]) -> Result<T, PulsrsError> {

        if self.is_scalar() {

            if !indices.is_empty() {
                return Err(
                    PulsrsError::TensorIndexingError { shape: self.shape().clone(), indices: indices.to_vec() }
                );
            }

            return Ok(self.buffer[0]);
        }
        
        let offset = self.stride.offset(indices)?;

        if offset >= self.buffer.len() {
            return Err(PulsrsError::TensorIndexingError { shape: self.shape().clone(), indices: indices.to_vec() });
        } else {
            Ok(
                self.buffer[offset]
            )
        }
    }

    pub fn flatten(&self) -> Result<Self, PulsrsError> {
        let numel = self.numel();
        self.reshape(vec![numel])
    }
}   

impl<T> From<T> for Tensor<T> 
where
    T: Copy + Default
{
    fn from(value: T) -> Self {
        Tensor::new_scalar(value)
    }
}

// later implement a from for nested vector, 
//where no shape passing is required just call the methods on a nested vector/list and a 
//tensor with inferred shape comes out. 
//Implement when moving to pyo3