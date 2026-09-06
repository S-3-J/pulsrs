use pulsars::tensor::Tensor;

fn main() {
    let t1 = Tensor::from_vector(
        (-784..784).map(|a| a as f32).collect::<Vec<f32>>(),
        vec![1, 2, 28, 28],
    )
    .unwrap();
    let t1 = t1.dim_names(
        ["batch", "channel", "height", "width"]
            .to_vec()
            .iter()
            .map(|a| a.to_string())
            .collect(),
    );

    print!("{}\n", t1);
}
