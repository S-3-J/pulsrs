use crate::cursor::{Cursor, IndexCursor, OffsetCursor};

pub enum IteratorBackend<'a> {
    Offset { start: usize, end: usize },

    Indexed { indexer: IndexCursor },

    StridedOffset { offsetter: OffsetCursor<'a> },
}

impl<'a> IteratorBackend<'a> {
    pub fn advance(&mut self) {
        match self {
            Self::Offset { start, end: _ } => *start += 1,

            Self::Indexed { indexer } => {
                indexer.advance();
            }

            Self::StridedOffset { offsetter } => {
                offsetter.advance();
            }
        }
    }
}
