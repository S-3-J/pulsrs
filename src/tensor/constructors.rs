use std::{ops::AddAssign};

use super::{Buffer, PulsrsError as PE, Rc, Shape, Stride, Tensor, Element};

use rand::{
    RngExt, 
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    rand_core::SeedableRng
};

impl<T: Element> Tensor<T> {
    // give vector and shape
    pub fn from_vector(data: Vec<T>, dims: Vec<usize>) -> Result<Self, PE> {
        let shape = Shape::new(dims);

        if shape.numel() != data.len() {
            return Err(PE::ShapeIncompatiblewithElements {
                expected: data.len(),
                permitted: shape.numel(),
            });
        }

        let buffer: Buffer<T> = Buffer::from_vec(data);
        let stride: Stride = Stride::from_shape(&shape);

        Ok(Tensor {
            buffer: Rc::new(buffer),
            shape,
            stride,
            offset: 0,
            contiguous: true,
            dtype: T::dtype(),
        })
    }

    pub fn new_scalar(data: T) -> Self {
        Tensor {
            buffer: Rc::new(Buffer::from_vec(vec![data])),
            shape: Shape::default(),
            stride: Stride::from(vec![]),
            offset: 0,
            contiguous: true,
            dtype: T::dtype(),
        }
    }

    pub fn zeros(dims: Vec<usize>) -> Option<Self> {
        let shape: Shape = Shape::new(dims);
        let stride: Stride = Stride::from_shape(&shape);

        Some(
            Tensor {
                buffer: Rc::new(Buffer::from_vec(vec![T::zero(); shape.numel()])),
                shape,
                stride,
                offset: 0,
                contiguous: true,
                dtype: T::dtype()
            }
        )
    }

    pub fn ones(dims: Vec<usize>) -> Option<Self> {
        let shape: Shape = Shape::new(dims);
        let stride: Stride = Stride::from_shape(&shape);

        Some(
            Tensor {
                buffer: Rc::new(Buffer::from_vec(vec![T::one(); shape.numel()])),
                shape,
                stride,
                offset: 0,
                contiguous: true,
                dtype: T::dtype()
            }
        )
    }

    pub fn random_init(dims: Vec<usize>, seed: u64) -> Self 
    where 
        StandardUniform: Distribution<T>
    {
        let shape: Shape = Shape::new(dims);
        let stride: Stride = Stride::from_shape(&shape);
        let mut data: Vec<T> = vec![T::zero(); shape.numel()];
        
        let mut rng = StdRng::seed_from_u64(seed);

        for i in &mut data {
            *i = rng.random();
        }

        Tensor{
            buffer: Rc::new(Buffer::from_vec(data)),
            shape,
            stride,
            offset: 0,
            contiguous: true,
            dtype: T::dtype()
        }
        
    }

    pub fn arange(start: T, stop: T, step: T) -> Self 
    where 
        T: PartialOrd + AddAssign<>
    {

        let mut data: Vec<T> = Vec::new();
        let mut val = start;
        data.push(val);

        while val < stop {
            val += step;
            data.push(val);
        }

        let shape: Shape = vec![data.len()].into();
        let stride = Stride::from_shape(&shape);

        Tensor {
            buffer : Rc::new(Buffer::from_vec(data)),
            shape,
            stride,
            offset: 0,
            contiguous: true,
            dtype: T::dtype()
        }
    }   

    pub fn dim_names(self, names: Vec<String>) -> Self {
        let shape: Shape = Shape::with_names(self.shape().dims().to_vec(), names);

        Tensor {
            buffer: self.buffer,
            shape,
            stride: self.stride,
            offset: 0,
            contiguous: self.contiguous,
            dtype: self.dtype,
        }
    }
}

// from a scalar
impl<T: Element> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Tensor::new_scalar(value)
    }
}
