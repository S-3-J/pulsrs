use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pulsrs::prelude::*;
use rand::seq::SliceRandom;
use std::hint::black_box as bb;

fn tensor_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Operations Benchmark");

    let sizes: Vec<f32> = vec![2.5, 12.5, 25.0, 125.0, 250.0, 2_500.0, 25_000.0, 250_000.0];

    for i in &sizes {
        let p = Tensor::<f32>::arange(0.0, *i, 0.25);
        let q = Tensor::<f32>::rev_arange(*i, 0.0, -0.25);

        let mut len = p.numel();
        let mut shape = Vec::<usize>::new();

        while len.is_multiple_of(2) {
            shape.push(2);
            len = len.div_ceil(2);
        }

        shape.push(len);

        let p_reshaped = match p.reshape(shape.clone()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        let q_reshaped = match q.reshape(shape.clone()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        let mut permuted_shape = (0..p_reshaped.ndim()).collect::<Vec<usize>>();
        permuted_shape.shuffle(&mut rand::rng());

        let p_permuted = match p_reshaped.permute(permuted_shape.clone()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        let q_permuted = match q_reshaped.permute(permuted_shape.clone()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        let p_small = match Tensor::<f32>::arange(0.0, *i, 0.5).reshape(shape[1..].to_vec()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        group.bench_with_input(
            BenchmarkId::new("Contiguous Addition Arange BM", i),
            i,
            |b, _i| b.iter(|| bb(&p + &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Addition Reshaped BM", i),
            i,
            |b, _i| b.iter(|| bb(&p_reshaped + &q_reshaped)),
        );

        group.bench_with_input(BenchmarkId::new("Permuted Addition BM", i), i, |b, _i| {
            b.iter(|| bb(&p_permuted + &q_permuted))
        });

        group.bench_with_input(
            BenchmarkId::new("Contiguous + Broadcasted BM", i),
            i,
            |b, _i| b.iter(|| bb(&p_reshaped + &p_small)),
        );
    }

    group.finish();
}

criterion_group!(benches, tensor_operations_benchmark);
criterion_main!(benches);
