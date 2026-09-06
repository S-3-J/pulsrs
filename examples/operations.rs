use std::time::Instant;

use pulsars::tensor::*;

fn main() {
    let t1 = Tensor::from_vector(
        (1..=1000000).map(|x| x as f32).collect::<Vec<f32>>(),
        vec![10, 100, 1000],
    )
    .unwrap();
    let t2 = t1.permute(vec![2, 0, 1]).unwrap();

    let start = Instant::now();

    for _ in 0..100 {
        let _ = t1.try_add(&t1).unwrap();
    }

    let time_t3 = start.elapsed();

    print!("Time for contiguous: {:?}\n", time_t3);
    print!("Amortized Time for contiguous: {:?}\n", time_t3 / 1000);

    for _ in 0..100 {
        let _ = t2.try_add(&t2).unwrap();
    }

    let time_t4 = start.elapsed();
    print!("Time for non contiguous: {:?}\n", time_t4 - time_t3);
    print!(
        "Amortized Time for contiguous: {:?}\n",
        (time_t4 - time_t3) / 1000
    );
    print!("Total: {:?}\n", time_t4)
}
