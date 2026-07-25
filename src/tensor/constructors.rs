
use super::{Tensor, Shape, Stride, Buffer, PulsrsError, Rc};

impl<T> Tensor<T> {
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
}

// from a scalar
impl<T> From<T> for Tensor<T> 
{
    fn from(value: T) -> Self {
        Tensor::new_scalar(value)
    }
}