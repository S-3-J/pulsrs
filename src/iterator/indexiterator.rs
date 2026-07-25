
use super::{IndexCursor, IndexIterator};

impl IndexIterator {
    pub fn new(cursor: IndexCursor) -> Self {
        Self {
            cursor,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor.finished() {
            None
        } else {
            let index = self.cursor.index().to_vec();
            self.cursor.advance();
            Some(index)
        }
    }
}

