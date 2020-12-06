use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate fourai;

use fourai::matrix::Matrix;

fn generate_sq(size: usize) -> Matrix<u32> {
    let mut val = vec![vec![0; size]; size];

    let possibles = [
        1123, 192, 999, 2, 1, 329, 1, 223, 2, 123, 456, 123, 75, 2, 5, 8, 1, 2, 3, 4, 5, 123214215,
        123, 24, 123, 1,
    ];

    for i in 0..size {
        for j in 0..size {
            val[i][j] = possibles[(size * i + j) % possibles.len()];
        }
    }

    Matrix::from(val)
}

fn criterion_benchmark(c: &mut Criterion) {
    let first_512 = generate_sq(256);
    let second_512 = first_512.clone();
    c.bench_function("matrix 512", |b| {
        b.iter(|| black_box(first_512.clone()) * black_box(second_512.clone()))
    });

    let first_256 = generate_sq(256);
    let second_256 = first_256.clone();
    c.bench_function("matrix 256", |b| {
        b.iter(|| black_box(first_256.clone()) * black_box(second_256.clone()))
    });

    let first_128 = generate_sq(128);
    let second_128 = first_128.clone();
    c.bench_function("matrix 128", |b| {
        b.iter(|| black_box(first_128.clone()) * black_box(second_128.clone()))
    });

    let first_64 = generate_sq(64);
    let second_64 = first_64.clone();
    c.bench_function("matrix 64", |b| {
        b.iter(|| black_box(first_64.clone()) * black_box(second_64.clone()))
    });

    let first_32 = generate_sq(32);
    let second_32 = first_32.clone();
    c.bench_function("matrix 32", |b| {
        b.iter(|| black_box(first_32.clone()) * black_box(second_32.clone()))
    });

    let first_16 = generate_sq(16);
    let second_16 = first_16.clone();
    c.bench_function("matrix 16", |b| {
        b.iter(|| black_box(first_16.clone()) * black_box(second_16.clone()))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
