use std::{collections::HashSet};
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

                if v.len() != dims {
                    panic!("Invalid custom priorities given for iteration style, not all axis present.")
                }

                let valid = v.iter()
                    .all(|a| a < &dims);

                if !valid {
                    panic!("Invalid custom priorities given for iteration style, dimensions exceed rank.")
                }

                let mut member: HashSet<usize> = HashSet::new();

                let unique = v.iter()
                    .all(|&a| member.insert(a));
                
                if !unique {
                    panic!("Invalid custom priorities given for iteration style, contains duplicate.")
                }
                v
            }
        }
    }

}


