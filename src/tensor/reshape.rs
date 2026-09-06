
use crate::{dtype::Element, function_signature};

use super::{Tensor, PulsrsError, Shape, Stride};

impl<T: Element> Tensor<T> 
where
    T: Copy + Default
{   
    pub fn view(&self, dims: Vec<usize>) -> Result<Self, PulsrsError> {
        
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
            if !self.is_contiguous() {
                Err(
                    PulsrsError::NonContiguousMemoryError { function_path: function_signature!() }
                )
            } else {
                let mut t = self.clone();
                t.shape = new_shape;
                t.stride = new_strides;

                Ok(
                    t
                )
            }

        }

    }

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

    pub fn flatten(&self) -> Result<Self, PulsrsError> {
        let numel = self.numel();
        self.reshape(vec![numel])
    }
}