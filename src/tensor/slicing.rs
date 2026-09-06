#![allow(clippy::manual_inspect)]
use std::rc::Rc;

use crate::{dtype::Element, error::PulsrsError as PE, shape::Shape, slicing::SliceRange, stride::Stride, tensor::Tensor};


impl<T: Element> Tensor<T> {
    pub fn slice(&self, ranges: Vec<Option<SliceRange>>) -> Result<Self, PE> {
        let dims = self.shape.dims();

        let not_none_axes = ranges.iter()
            .filter(|r| r.is_some())
            .count();

        if dims.len() != not_none_axes {
            return Err(
                PE::RankMismatch { expected: dims.len(), found: not_none_axes }
            )
        }

        let mut i = 0usize;

        let processed_ranges = ranges
            .iter()
            .map(
                |&sr| if let Some( r ) = sr {
                    r.resolve(0, dims[i]).map(|m| {
                        i+=1;
                        m
                    })
                } else {
                    Ok((1, 1, 0, true)) 
                }
            )
            .collect::<Result<Vec<_>, _>>()?;
         
        //print!("Processed Ranges: {:?}", processed_ranges);

        let slice_offset = self.stride.offset(
            &processed_ranges.iter()
                .filter(|a| !a.3)
                .map(|a| if a.2.is_negative() {a.1} else {a.0})
                .collect::<Vec<_>>()
        )?;

        let slice_shape = Shape::new(
            processed_ranges.iter().map(|a| if !a.3 {((a.1 - a.0) / a.2.unsigned_abs()) + 1} else {1}).collect()
        );

        //print!("Slice Shape: {:?}", slice_shape);
        
        let strides = self.strides();
        i = 0;
        
        let slice_stride = Stride::from_vec(
            processed_ranges.iter()
            .map(|s| if !s.3 {
                let st = strides[i] * s.2;
                i+=1;
                st
            } else {
                0
            })
            .collect()
        );
        //print!("Slice Stride: {:?}", slice_stride);

        Ok(
            Self {
                buffer: Rc::clone(&self.buffer),
                shape: slice_shape,
                stride: slice_stride,
                offset: slice_offset as usize,
                contiguous: false,
                dtype: T::dtype()
            }
        )
    }
}