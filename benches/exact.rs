use bernoulli::BernoulliExact;
use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, Criterion,
};
use rand::{
    distributions::{Bernoulli, Distribution},
    rngs::StdRng,
    Rng, SeedableRng,
};
use rand_pcg::Pcg64;

const ITERATIONS: usize = 1;
const USE_EXACT: bool = true;

fn bool_test<R: Rng, D: Distribution<bool>>(rng: &mut R, dist: &D) {
    for _ in 0..black_box(ITERATIONS) {
        black_box(dist.sample(rng));
    }
}

fn bench_new<M: Measurement, D: Distribution<bool>, F: Fn(u32, u32) -> D>(
    c: &mut BenchmarkGroup<M>,
    n: u32,
    d: u32,
    f: F,
) {
    c.bench_function(format!("new {}/{}", n, d).as_str(), |b| {
        b.iter(|| black_box(f(black_box(n), black_box(d))))
    });
}

fn bench_bool_test<M: Measurement, D: Distribution<bool>, F: Fn(u32, u32) -> D, R: Rng>(
    c: &mut BenchmarkGroup<M>,
    n: u32,
    d: u32,
    f: F,
    rng: &mut R,
) {
    let dist = f(n, d);
    c.bench_function(format!("ratio {}/{}", n, d).as_str(), |b| {
        b.iter(|| bool_test(black_box(rng), black_box(&dist)))
    });
}

fn bench_set<D: Distribution<bool>, F: Fn(u32, u32) -> D, R: Rng>(
    c: &mut Criterion,
    name: &str,
    f: F,
    mut rng: R,
) {
    let mut c = c.benchmark_group(name);
    let values = [(10, 15), (5, 10), (5, 50000), (50000, 50000), (0, 50000)];

    for (n, d) in values {
        bench_new(&mut c, n, d, &f);
    }

    for (n, d) in values {
        bench_bool_test(&mut c, n, d, &f, &mut rng);
    }

    c.finish();
}

fn bench<R: Rng>(c: &mut Criterion, name: &str, rng: R) {
    if USE_EXACT {
        bench_set(
            c,
            name,
            |n, d| BernoulliExact::from_ratio(n, d).unwrap(),
            rng,
        )
    } else {
        bench_set(c, name, |n, d| Bernoulli::from_ratio(n, d).unwrap(), rng)
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = StdRng::from_entropy();
    let rng2 = Pcg64::new(rng.gen(), 0);
    bench(c, "StdRng", rng);
    bench(c, "Pcg64", rng2);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
