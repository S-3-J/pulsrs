
pub enum IterStyle{
    Cstyle,
    Fstyle,
    Custom(Vec<usize>),
}

impl IterStyle {

    fn c_style(dims: usize) -> Vec<usize> {
        (0..dims).rev().collect::<Vec<usize>>()
    }

    fn f_style(dims: usize) -> Vec<usize> {
        (0..dims).collect::<Vec<usize>>()
    }

    pub fn process(self, dims: usize) -> Vec<usize> {
        match self {
            Self::Cstyle => {
                Self::c_style(dims)
            },

            Self::Fstyle => {
                Self::f_style(dims)
            },

            Self::Custom(v) => {
                v
            }
        }
    }

}


