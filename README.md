# Bernoulli

A generic replacement for [`rand::distributions::Bernoulli`](https://docs.rs/rand/latest/rand/distributions/struct.Bernoulli.html) which uses exact precision instead of a 64-bit approximation, `bernoulli::BernoulliExact<T>` provides an exact distribution for any rational value representable by `T`.

`BernoulliExact<u32>` performs on-par with `Bernoulli`, though with a non-trivial memory overhead.

| Type                  | Size       |
|-----------------------|------------|
| `Bernoulli`           | `8` bytes  |
| `BernoulliExact<u32>` | `20` bytes |
| `BernoulliExact<u8>`  | `5` bytes  |