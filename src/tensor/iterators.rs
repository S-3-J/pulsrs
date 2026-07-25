use super::Tensor;
use crate::{
    cursor::IndexCursor, 
    iterator::{
        IndexIterator, 
        TensorIterator, 
        iteration_style::*
    }
};

impl<T> Tensor<T> {

    pub fn index_iterator(&self, style: IterStyle) -> IndexIterator {

        let priority: Vec<usize> = style.process(self.ndim());

        IndexIterator::new(
            IndexCursor::new(self.shape().dims().to_vec(), priority)
        )
    }

    pub fn tensor_iterator<'a>(&'a self, style: IterStyle) -> TensorIterator<'a, T>
    where
        T: 'a
    {
        let priority: Vec<usize> = style.process(self.ndim());
        
        TensorIterator::new(
            self,
            IndexCursor::new(
                self.shape().dims().to_vec(),
                priority,
            )
        )
    }
}