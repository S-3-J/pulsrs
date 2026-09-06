use super::Tensor;
use crate::{
    cursor::IndexCursor, dtype::Element, iterator::{
        IndexIterator, 
        TensorIterator, 
        iteration_style::*
    }
};

impl<T: Element> Tensor<T> {

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
        TensorIterator::new(
            self,
            style,
        )
    }
}