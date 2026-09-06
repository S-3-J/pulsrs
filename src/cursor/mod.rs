
pub mod index_cursor;
pub mod offset_cursor;

pub trait Cursor{
    type Output;

    fn advance(&mut self);

    fn reset(&mut self);
    
    fn finished(&self) -> bool;
    
    fn has_elements(&self) -> bool;

    fn current(&self) -> Self::Output;

}

pub struct IndexCursor {
    dims: Vec<usize>,
    current_index: Vec<usize>,
    priority: Vec<usize>,
    overflow: bool,
    filled: bool,
}

pub struct OffsetCursor<'a> {
    dims: &'a [usize],
    strides: &'a [isize],
    priority: Vec<usize>,
    current_index: Vec<usize>,
    offset: isize,
    overflow: bool,
    filled: bool,
}