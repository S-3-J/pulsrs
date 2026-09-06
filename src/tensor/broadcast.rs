
use crate::{dtype::Element, shape::Shape, stride::Stride, tensor::Tensor};


impl<T: Element> Tensor<T> {
    pub fn broadcast(&self, shape: &[usize]) -> Self {
        let dims = self.shape().dims();
        let strides = self.strides();

        let mut new_strides = vec![0isize; shape.len()];

        let rank_diff = shape.len() - dims.len();

        for i in 0..dims.len() {
            let src_dim = dims[i];
            let dst_dim = shape[rank_diff + i];

            if src_dim == dst_dim {
                new_strides[rank_diff + i] = strides[i];
            } else if src_dim == 1 {
                // Broadcasted dimension → zero stride
                new_strides[rank_diff + i] = 0;
            }
        }

        Tensor {
            buffer: self.buffer.clone(),
            shape: Shape::new(shape.to_vec()),
            stride: Stride::from_vec(new_strides),
            offset: self.offset,
            contiguous: false,
            dtype: self.dtype,
        }
    }

    pub fn is_broadcastible_for(&self, other_shape: &[usize]) -> (bool, Vec<usize>)
    {
        let this_shape = self.shape().dims();

        let mut final_shape = Vec::with_capacity(this_shape.len().max(other_shape.len()));

        let mut this = this_shape.iter().rev();
        let mut other = other_shape.iter().rev();

        loop {
            match (this.next(), other.next()) {
                (Some(&i), Some(&j)) => {
                    match (i, j) {
                        (i, j) if i == j => final_shape.push(i),
                        (1, j) => final_shape.push(j),
                        (i, 1) => final_shape.push(i),
                        _ => return (false, vec![]),
                    }
                }

                (Some(&i), None) => final_shape.push(i),
                (None, Some(&j)) => final_shape.push(j),
                (None, None) => break,
            }
        }

        final_shape.reverse();

        (true, final_shape)
    }

    /*
    pub fn try_broadcast(&self, other_shape: &[usize]) -> Result<Self, PulsrsError> {
        let (safe, new_shape) = self.is_broadcastible_for(other_shape);
        
        if safe {
            Ok(
                self.broadcast(other_shape)
            )
        } else {
            Err(
                
            )
        }
    }
*/
}
