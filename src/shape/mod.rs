pub mod shape_methods;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
    names: Option<Vec<String>>,
}
