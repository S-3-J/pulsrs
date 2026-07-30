
use std::fmt::{Display, Write};
use std::any::type_name;

use crate::tensor::Tensor;

impl<T> Display for Tensor<T> 
where 
    T: Copy + Display
{
    
    fn fmt(
        &self, 
        f: &mut std::fmt::Formatter<'_>
    ) -> std::fmt::Result 
    {
        fn recursive_printer<T>(t: &Tensor<T>, depth: usize, index: &mut Vec<usize>, dims: &[usize]) -> String
        where 
            T: Copy + Display
        {
            if depth >= dims.len() {
                return "".to_string();
            }

            let mut master_string = String::new();

            for i in 0..dims[depth] {
                index[depth] = i;
                let s = recursive_printer(t, depth+1, index, dims);

                if s.is_empty() {
                    let mut a = t.get(index).unwrap().to_string();
                    if i < dims[depth] - 1 {
                        a.push_str(", ");
                    }
                    let _ = write!(
                        &mut master_string,
                        "{}",
                        a
                    );
                } else {
                    let _ = write!(
                        &mut master_string,
                        "{}",
                        s
                    );
                }
            }
            let space = vec!["    "; depth].join("");

            master_string = "\n".to_string() + &space + "[" + &master_string + {
                if depth + 1 != dims.len() {
                    &space
                } else {
                    ""
                }
            } + "]\n" ;

            master_string 
        }

        if self.is_scalar() {
            write!(
                f,
                "[{}], dtype: {}",
                self.get_from_offset(0).unwrap(),
                type_name::<T>()
            )
        } else {
            let mut index = vec![0usize; self.ndim()];
            let dims = self.shape().dims();
            let t = recursive_printer(self, 0, &mut index, dims);

            write!(
                f,
                "{}, dtype: {}, shape: {}",
                t,
                type_name::<T>(),
                self.shape().clone()
            )
        }

    }
}