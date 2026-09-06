use super::{Cursor, OffsetCursor};

// special fast cursor just for contiguous tensors
impl<'a> OffsetCursor<'a> {
    pub fn new(dims: &'a [usize], strides: &'a [isize], priority: Vec<usize>, offset: isize) -> Self {
        let filled = dims.iter().product::<usize>() != 0usize;

        Self {
            dims,
            strides,
            priority,
            current_index: vec![0usize; dims.len()],
            offset,
            overflow: false,
            filled,
        }
    }
}

impl<'a> Cursor for OffsetCursor<'a> {
    type Output = usize;

    #[inline(always)]
    fn advance(&mut self) {
        if self.dims.is_empty() {
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
                self.offset -= self.strides[axis] * self.current_index[axis] as isize;
                self.current_index[axis] = 0;
            }
        }
    }

    #[inline(always)]
    fn current(&self) -> Self::Output {
        self.offset as usize
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
