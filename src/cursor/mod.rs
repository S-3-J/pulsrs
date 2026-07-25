
#[allow(dead_code)]
pub struct IndexCursor {
    dims: Vec<usize>,
    current_index: Vec<usize>,
    priority: Vec<usize>,
    overflow: bool,
}

#[allow(dead_code)]
impl IndexCursor {
    pub fn new(dims: Vec<usize>, priority: Vec<usize>) -> Self {
        let d_len = dims.len();
        assert!(d_len == priority.len(), "Index Cursor got mismatching dimensions and priorities");
        Self {
            dims,
            current_index: vec![0; d_len],
            priority,
            overflow: false,
        }
    }

    pub fn advance(&mut self) {

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

    pub fn index(&self) -> &[usize] {
        &self.current_index
    }

    pub fn reset(&mut self) {
        self.current_index.fill(0);
        self.overflow = false;
    }

    pub fn finished(&self) -> bool{
        self.overflow
    }
}
