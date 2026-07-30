use super::{Tensor, Buffer, Stride, Rc};

impl<T> Tensor<T> 
where 
    T: Copy + Default
{
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
                for i in 0..dims[depth] {
                    index[depth] = i;
                    if depth+1 == dims.len() {
                        // should panic and crash if erroneous; 
                        //however its not possible to crash here as the tensor won't index out of bounds.
                        let val = t.get(index).unwrap();
                        let new_offset = new_stride.offset(&index).unwrap();
                        
                        new_buffer[new_offset] = val;   
                    } else {
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
}