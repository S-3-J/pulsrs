use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pulsrs::prelude::*;
use std::hint::black_box as bb;

fn c_t_o_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Contiguous Tensor Operations");
    let sizes: Vec<usize> = vec![100_000, 250_000, 1_000_000];

    // corresponds to 500K, 1M and 2M elements.
    let steps: Vec<f32> = vec![0.2, 0.25, 0.5];

    for (&size, &step) in sizes.iter().zip(steps.iter()) {
        // initialize tensors
        let p = Tensor::<f32>::arange(0.0, size as f32, step);
        let q = Tensor::<f32>::rev_arange(size as f32, 0.0, -step);

        // build shape
        let mut len = p.numel();
        let mut shape = Vec::<usize>::new();

        while len.is_multiple_of(2) {
            shape.push(2);
            len = len.div_ceil(2);
        }

        shape.push(len);

        // reshape them
        let p_reshaped = match p.reshape(shape.clone()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        let q_reshaped = match q.reshape(shape.clone()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        group.bench_with_input(
            BenchmarkId::new("Contiguous Addition Arange BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p + &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Addition Reshaped BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p_reshaped + &q_reshaped)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Subtraction Arange BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p - &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Subtraction Reshaped BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p_reshaped - &q_reshaped)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Multiplication Arange BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p * &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Multiplication Reshaped BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p_reshaped * &q_reshaped)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Division Arange BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p / &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Division Reshaped BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p_reshaped / &q_reshaped)),
        );
    }

    group.finish();
}

fn c_t_o_benchmark_64(c: &mut Criterion) {
    let mut group = c.benchmark_group("Contiguous Tensor Operations f64");
    let sizes: Vec<usize> = vec![100_000, 250_000, 1_000_000];

    // corresponds to 500K, 1M and 2M elements.
    let steps: Vec<f64> = vec![0.2, 0.25, 0.5];

    for (&size, &step) in sizes.iter().zip(steps.iter()) {
        // initialize tensors
        let p = Tensor::<f64>::arange(0.0, size as f64, step);
        let q = Tensor::<f64>::rev_arange(size as f64, 0.0, -step);

        // build shape
        let mut len = p.numel();
        let mut shape = Vec::<usize>::new();

        while len.is_multiple_of(2) {
            shape.push(2);
            len = len.div_ceil(2);
        }

        shape.push(len);

        // reshape them
        let p_reshaped = match p.reshape(shape.clone()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        let q_reshaped = match q.reshape(shape.clone()) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        group.bench_with_input(
            BenchmarkId::new("Contiguous Addition Arange BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p + &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Addition Reshaped BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p_reshaped + &q_reshaped)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Subtraction Arange BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p - &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Subtraction Reshaped BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p_reshaped - &q_reshaped)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Multiplication Arange BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p * &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Multiplication Reshaped BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p_reshaped * &q_reshaped)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Division Arange BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p / &q)),
        );

        group.bench_with_input(
            BenchmarkId::new("Contiguous Division Reshaped BM", size),
            &size,
            |b, _i| b.iter(|| bb(&p_reshaped / &q_reshaped)),
        );
    }

    group.finish();
}

criterion_group!(benches2, c_t_o_benchmark, c_t_o_benchmark_64);
criterion_main!(benches2);
