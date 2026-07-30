
use super::{IndexCursor, IndexIterator};
use crate::cursor::Cursor;

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
        if !self.cursor.has_elements() || self.cursor.finished() {
            None
        } else {
            let index = self.cursor.current();
            self.cursor.advance();
            Some(index)
        }
    }
}

