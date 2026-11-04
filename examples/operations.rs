use std::time::Instant;

use pulsrs::tensor::*;

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

    println!("Time for contiguous: {:?}", time_t3);
    println!("Amortized Time for contiguous: {:?}", time_t3 / 1000);

    for _ in 0..100 {
        let _ = t2.try_add(&t2).unwrap();
    }

    let time_t4 = start.elapsed();
    println!("Time for non contiguous: {:?}", time_t4 - time_t3);
    println!(
        "Amortized Time for contiguous: {:?}",
        (time_t4 - time_t3) / 1000
    );
    println!("Total: {:?}", time_t4)
}
