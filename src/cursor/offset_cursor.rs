
use super::{
    OffsetCursor,
    Cursor
};

// special fast cursor just for contiguous tensors
impl<'a> OffsetCursor<'a> {
    pub fn new(
        dims: &'a [usize], 
        strides: &'a [usize],
        priority: Vec<usize>,
    ) -> Self {

        let filled = dims.iter().product::<usize>() != 0usize ;

        Self {
            dims,
            strides,
            priority,
            current_index: vec![0usize; dims.len()],
            offset: 0,
            overflow: false,
            filled,
        }
    }
}

impl<'a> Cursor for OffsetCursor<'a> {
    type Output = usize;

    #[inline(always)]
    fn advance(&mut self) {
        if self.dims.len() == 0 {
            self.overflow = true;
        } else {
            self.overflow = true;
            for &axis in self.priority.iter() {
                if self.current_index[axis] + 1 < self.dims[axis] {
                    self.current_index[axis] += 1;
                    self.offset += self.strides[axis];
                    self.overflow = false;
                    break;
                }
                self.offset -= self.strides[axis] * self.current_index[axis];
                self.current_index[axis] = 0;
            }
        }
    }

    #[inline(always)]
    fn current(&self) -> Self::Output {
        self.offset
    }

    fn reset(&mut self) {
        self.current_index.fill(0);
        self.overflow = false;
        self.offset = 0;
    }

    fn finished(&self) -> bool {
        self.overflow
    }

    fn has_elements(&self) -> bool {
        self.filled
    }
}