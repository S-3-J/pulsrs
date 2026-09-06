
use pulsars::prelude::*;

fn main() {

    let t2 = Tensor::<i8>::arange(1, 30, 1).reshape(vec![5, 2, 3]).expect("Some bruh!");
    let t3 = t2.slice(
        vec![(..).iosr(), None, (..).iosr(), (..).iosr()]
    ).unwrap().broadcast(&[2, 5, 2, 2, 3]);

    print!("{}\n\n", t2);
    print!("{}\n\n", t3);
}