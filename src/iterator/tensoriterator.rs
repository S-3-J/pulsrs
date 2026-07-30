
use std::{fmt::Debug};
use crate::cursor::OffsetCursor;
use crate::iterator::iterator_backend::IteratorBackend;
use crate::{cursor::{Cursor}, tensor::Tensor};
use crate::iterator::iteration_style::IterStyle;

use super::TensorIterator;

impl<'a, T> TensorIterator<'a, T>
where
    T: 'a
{
    pub fn new(tensor: &'a Tensor<T>, priority: IterStyle) -> Self 
    {
        let backend = match tensor.is_contiguous() && matches!(priority, IterStyle::Cstyle) {
            true => {
                IteratorBackend::Offset { start: 0, end: tensor.numel() }
            },
            // defaults to strided offset
            false => {
                IteratorBackend::StridedOffset {
                    offsetter: OffsetCursor::new(tensor.shape().dims(), tensor.strides(), priority.process(tensor.ndim()))
                }
            }
        };

        Self {
            tensor,
            backend
        }
    }

    pub fn backend(&self) -> &IteratorBackend<'a> {
        &self.backend
    }
}


impl<'a, T> Iterator for TensorIterator<'a, T>
where
    T: 'a + Copy + Debug
{
    type Item = T;
    
    fn next(&mut self) -> Option<Self::Item> {

        match &self.backend {
            IteratorBackend::Offset { start, end } => {
                if start == end {
                    None
                } else {
                    let e = self.tensor.get_from_offset(*start).unwrap();
                    self.backend.advance();
                    Some(e)
                }
            },

            // older slow indexing via stride calculation
            IteratorBackend::Indexed { indexer } => {
                if !indexer.has_elements() || indexer.finished() {
                    None
                } else {
                    let index = indexer.current();
                    let e = self.tensor.get(&index).unwrap();
                    self.backend.advance();

                    Some(e)
                }
            },

            IteratorBackend::StridedOffset { offsetter } => {
                if !offsetter.has_elements() || offsetter.finished() {
                    None
                } else {
                    let offset = offsetter.current();
                    let e = self.tensor.get_from_offset(offset).unwrap();
                    self.backend.advance();

                    Some(e)
                }
            }
        }
        
    }
}