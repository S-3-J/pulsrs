
use pulsars::tensor::*;

fn main() {
    let t1 = Tensor::from_vector((1..=24).collect::<Vec<i32>>(), vec![2, 3, 4]).unwrap();

    let t2 = t1.reduce_sum(&[1]);
    let t3 = t1.reduce_max(&[1]);
    let t4 = (&t1).reduce_mean(&[1]);


    print!("{}\n", t1);
    print!("{}\n", t2);
    print!("{}\n", t3);
    print!("{}\n", t4);
}