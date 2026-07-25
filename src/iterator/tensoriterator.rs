
use std::fmt::Debug;
use crate::{cursor::IndexCursor, tensor::Tensor};

use super::TensorIterator;

impl<'a, T> TensorIterator<'a, T>
where
    T: 'a
{
    pub fn new(tensor: &'a Tensor<T>, cursor: IndexCursor) -> Self {
        Self {
            tensor,
            cursor
        }
    }
}


impl<'a, T> Iterator for TensorIterator<'a, T>
where
    T: 'a + Copy + Debug
{
    type Item = T;
    
    fn next(&mut self) -> Option<Self::Item> {

        if self.cursor.finished() {
            None
        } else {
            let index = self.cursor.index();
            let e = self.tensor
                .get(index)
                .expect("Index Cursor, produced an invalid index. This should not happen, pray to gods atp!");
    
            self.cursor.advance();
    
            Some(e)
        }

        
    }
}