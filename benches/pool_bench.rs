use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate fourai;

use fourai::ai::pool::{Pool, PoolProperties};
use fourai::ai::NNPlayer;
use fourai::{ai::nn::Activation, pool_props};

fn gen_props(size: usize) -> PoolProperties {
    pool_props! {
        surviving_amount => size,
        mutation_amount => 3,
        mutation_range => 0.05,
        crossover_amount => 1,
        structure => vec![42, 98, 98, 98, 7],
        activations => vec![
            Activation::Sigmoid,
            Activation::Sigmoid,
            Activation::Sigmoid,
            Activation::Sigmoid,
        ],
        generations => 1,
        save_interval => 100000,
        compare_interval => 100000,
        file_path => std::path::PathBuf::from("dummy_path/")
    }
}
fn big_bench(c: &mut Criterion) {
    let pool: Pool<NNPlayer> = Pool::new(gen_props(10));
    c.bench_function("run pool, 10 surviving", |b| {
        b.iter(|| black_box(pool.clone().training_loop(0)));
    });
}

fn small_bench(c: &mut Criterion) {
    let pool_small: Pool<NNPlayer> = Pool::new(gen_props(5));
    c.bench_function("run pool, 5 surviving", |b| {
        b.iter(|| black_box(pool_small.clone().training_loop(0)));
    });
}

criterion_group! {
    name = small_benches;
    config = Criterion::default().sample_size(10);
    targets = small_bench
}
criterion_group! {
    name = big_benches;
    config = Criterion::default().sample_size(10);
    targets = big_bench
}
criterion_main!(small_benches, big_benches);
