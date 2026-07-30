
use super::{IndexCursor, Cursor};

impl IndexCursor {
    pub fn new(dims: Vec<usize>, priority: Vec<usize>) -> Self {
        let d_len = dims.len();
        assert!(d_len == priority.len(), "Index Cursor got mismatching dimensions and priorities");
        
        let filled = dims.iter().product::<usize>() > 0;
        
        Self {
            dims,
            current_index: vec![0; d_len],
            priority,
            overflow: false,
            filled,
        }
    }

    pub fn contiguous_style(&self) -> bool {
        let prs = self.priority.windows(2)
            .map(|w| w[0] > w[1])
            .fold(true, |acc, e| acc && e);
        
        prs
    }
}

impl Cursor for IndexCursor {
    type Output = Vec<usize>;

    fn advance(&mut self) {

        if self.dims.len() == 0 {
            self.overflow = true;
        } else {
            self.overflow = true;
            for &dim in self.priority.iter() {
                if self.current_index[dim] + 1 < self.dims[dim] {
                    self.current_index[dim] += 1;
                    self.overflow = false;
                    break
                } 
                self.current_index[dim] = 0;
            }
        }

    }

    fn current(&self) -> Self::Output {
        self.current_index.to_vec()
    }

    fn reset(&mut self) {
        self.current_index.fill(0);
        self.overflow = false;
    }

    fn finished(&self) -> bool{
        self.overflow
    }

    fn has_elements(&self) -> bool {
        self.filled
    }

}
