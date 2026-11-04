use pulsrs::iterator::iteration_style::IterStyle;
use pulsrs::prelude::*;

fn main() {
    let t1 = Tensor::<i16>::arange(1, 10, 1)
        .reshape(vec![3, 3])
        .unwrap()
        .slice(vec![(.., -1).iosr(), (..).iosr()]);

    if let Ok(x) = t1 {
        let a = x.tensor_iterator(IterStyle::Fstyle).collect::<Vec<_>>();
        print!("{:?}", a);
    } else {
        panic!("{}", t1.unwrap_err())
    }
}
