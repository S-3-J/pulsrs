use criterion::{Criterion, criterion_group, criterion_main};
use std::{hint::black_box as bb, panic};

use pulsrs::prelude::*;

fn tensor_reshape_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Reshape Benchmarks on f32 Tensors");

    let wrapped_p = Tensor::<f32>::arange(1.0, 250_001.0, 0.25).reshape(vec![5, 20, 25, 4, 100]);

    let Ok(p) = wrapped_p else {
        panic!("{}", wrapped_p.unwrap_err())
    };

    let wrapped_q = p.permute(vec![3, 1, 4, 0, 2]);

    let Ok(q) = wrapped_q else {
        panic!("{}", wrapped_q.unwrap_err())
    };

    group.bench_function("Reshape on Contiguous", |b| {
        b.iter(|| bb(p.reshape(vec![5, 5, 100, 20, 20])))
    });

    group.bench_function("Reshape on Non-Contiguous", |b| {
        b.iter(|| bb(q.reshape(vec![5, 5, 100, 20, 20])))
    });

    group.finish();
}

criterion_group!(reshape_bench, tensor_reshape_benchmark);
criterion_main!(reshape_bench);
