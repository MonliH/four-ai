use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[macro_use]
extern crate fourai;

use fourai::ai::matrix::Matrix;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("matrix small", |b| b.iter(|| black_box(mat![2391, 232; 32321, 23; 231, 323]) * black_box(mat![2391, 232, 32321; 23, 23, 323])));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
