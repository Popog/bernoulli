#![no_std]

use num_traits::Zero;
use rand::{
    distributions::{uniform::SampleUniform, BernoulliError, Distribution, Uniform},
    Rng,
};
use core::{
    cmp::Ordering,
    fmt,
    ops::{Add, Sub},
};

/// A trait encapsulating the generic requirements of `BernoulliExact`
pub trait Ratio:
    Sized
    + Zero
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + Ord
    + SampleUniform
{
}

impl<
        T: Sized
            + Zero
            + for<'a> Add<&'a Self, Output = Self>
            + for<'a> Sub<&'a Self, Output = Self>
            + Ord
            + SampleUniform,
    > Ratio for T
{
}

enum BernoulliInternal<T: Ratio> {
    False,
    True,
    Dist { threshold: T, dist: Uniform<T> },
}

impl<T: Ratio + Clone> Clone for BernoulliInternal<T>
where
    T::Sampler: Clone,
{
    fn clone(&self) -> Self {
        use BernoulliInternal::*;
        match self {
            False => False,
            True => True,
            Dist { threshold, dist } => Dist {
                threshold: threshold.clone(),
                dist: dist.clone(),
            },
        }
    }
}

///
/// This is a special case of the Binomial distribution where `n = 1`.
///
/// # Example
///
/// ```rust
/// use rand::distributions::Distribution;
/// use bernoulli::BernoulliExact;
///
/// let d = BernoulliExact::from_true_false(3, 10).unwrap();
/// let v = d.sample(&mut rand::thread_rng());
/// println!("{} is from a Bernoulli distribution", v);
/// #
/// # // Check that the sizes match the docs below
/// # use std::mem::size_of;
/// # assert_eq!(size_of::<rand::distributions::Bernoulli>(), 8);
/// # assert_eq!(size_of::<bernoulli::BernoulliExact<u32>>(), 20);
/// # assert_eq!(size_of::<bernoulli::BernoulliExact<u8>>(), 5);
/// ```
///
/// # Precision
///
/// The `BernoulliExact` distribution uses exact integer precision so any rational probability of `T`s can be accurately sampled.
///
/// # `Bernoulli` Comparison
///
/// `BernoulliExact` performs similarly to [`Bernoulli`](rand::distributions::Bernoulli), but generally uses more space.
/// 
/// | Type                  | Size       |
/// |-----------------------|------------|
/// | `Bernoulli`           | `8` bytes  |
/// | `BernoulliExact<u32>` | `20` bytes |
/// | `BernoulliExact<u8>`  | `5` bytes  |
pub struct BernoulliExact<T: Ratio> {
    internal: BernoulliInternal<T>,
}

impl<T: Ratio + Clone> Clone for BernoulliExact<T>
where
    T::Sampler: Clone,
{
    fn clone(&self) -> Self {
        BernoulliExact {
            internal: self.internal.clone(),
        }
    }
}

impl<T: Ratio + fmt::Debug> fmt::Debug for BernoulliExact<T>
where
    T::Sampler: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use BernoulliInternal::*;
        let mut f = f.debug_struct("BernoulliExact");
        match &self.internal {
            False => f.field("value", &false),
            True => f.field("value", &true),
            Dist { threshold, dist } => f.field("threshold", threshold).field("dist", dist),
        }
        .finish()
    }
}

impl<T: Ratio> BernoulliExact<T> {
    /// Construct a new `BernoulliExact` with the probability ratio of `t` to `f`.
    ///
    /// `from_true_false(2, 1)` will return a `BernoulliExact` with a 2-in-3 chance (about 67%) of returning `true` and a 1-in-3 chance (about 33%) of returning `false`.
    pub fn from_true_false(t: T, f: T) -> Result<Self, BernoulliError> {
        use Ordering::*;

        let zero = T::zero();
        match (t.cmp(&zero), f.cmp(&zero)) {
            (Equal, Equal) => Err(BernoulliError::InvalidProbability),
            // t == 0
            (Equal, _) => Ok(BernoulliExact {
                internal: BernoulliInternal::False,
            }),
            // f == 0
            (_, Equal) => Ok(BernoulliExact {
                internal: BernoulliInternal::True,
            }),
            // Negative numbers
            (Less, Less) => Ok(BernoulliExact {
                internal: BernoulliInternal::Dist {
                    dist: Uniform::new(t + &f, zero),
                    threshold: f,
                },
            }),
            // Positive numbers
            (Greater, Greater) => Ok(BernoulliExact {
                internal: BernoulliInternal::Dist {
                    dist: Uniform::new(zero, f + &t),
                    threshold: t,
                },
            }),
            _ => Err(BernoulliError::InvalidProbability),
        }
    }

    /// Construct a new `BernoulliExact` with the probability ratio of `n` over `d`.
    ///
    /// `from_ratio(2, 3)` will return a `BernoulliExact` with a 2-in-3 chance (about 67%) of returning `true` and a 1-in-3 chance (about 33%) of returning `false`.
    pub fn from_ratio(n: T, d: T) -> Result<Self, BernoulliError> {
        use Ordering::*;

        let zero = T::zero();
        match (n.cmp(&zero), d.cmp(&zero), n.cmp(&d)) {
            (_, Equal, _) => Err(BernoulliError::InvalidProbability),
            // n == 0
            (Equal, _, _) => Ok(BernoulliExact {
                internal: BernoulliInternal::False,
            }),
            // n == d
            (_, _, Equal) => Ok(BernoulliExact {
                internal: BernoulliInternal::True,
            }),
            (Greater, Greater, Greater) | (Less, Less, Less) => {
                Err(BernoulliError::InvalidProbability)
            }
            // Negative numbers
            (Less, Less, Greater) => Ok(BernoulliExact {
                internal: BernoulliInternal::Dist {
                    threshold: T::zero() - &n + &d,
                    dist: Uniform::new(d, zero),
                },
            }),
            // Positive numbers
            (Greater, Greater, Less) => Ok(BernoulliExact {
                internal: BernoulliInternal::Dist {
                    threshold: n,
                    dist: Uniform::new(zero, d),
                },
            }),
            _ => Err(BernoulliError::InvalidProbability),
        }
    }
}

impl<T: Ratio> Distribution<bool> for BernoulliExact<T> {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        use BernoulliInternal::*;
        match &self.internal {
            True => true,
            False => false,
            Dist { dist, threshold } => dist.sample(rng) < *threshold,
        }
    }
}

#[cfg(test)]
mod test {
    use super::{BernoulliExact, Ratio};
    use rand::{distributions::Distribution, Error, Rng, RngCore};
    use rand_pcg::Pcg64;

    #[test]
    fn test_skip_rng() {
        // We prefer to be explicit here.
        #![allow(clippy::bool_assert_comparison)]

        struct NoRng;
        impl RngCore for NoRng {
            fn next_u32(&mut self) -> u32 {
                unimplemented!()
            }
            fn next_u64(&mut self) -> u64 {
                unimplemented!()
            }
            fn fill_bytes(&mut self, _: &mut [u8]) {
                unimplemented!()
            }
            fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Error> {
                unimplemented!()
            }
        }

        fn from_true_false<T: Ratio>(t: T, f: T) -> BernoulliExact<T> {
            BernoulliExact::from_true_false(t, f)
                .expect("failed to create BernoulliExact from ratio")
        }
        fn from_ratio<T: Ratio>(n: T, d: T) -> BernoulliExact<T> {
            BernoulliExact::from_ratio(n, d).expect("failed to create from fraction")
        }

        let mut r = NoRng;
        assert_eq!(r.sample::<bool, _>(&from_true_false(0, 1)), false);
        assert_eq!(r.sample::<bool, _>(&from_ratio(0, 1)), false);
        assert_eq!(r.sample::<bool, _>(&from_true_false(1, 0)), true);
        assert_eq!(r.sample::<bool, _>(&from_ratio(1, 1)), true);

        assert_eq!(r.sample::<bool, _>(&from_true_false(0, -1)), false);
        assert_eq!(r.sample::<bool, _>(&from_ratio(0, -1)), false);
        assert_eq!(r.sample::<bool, _>(&from_true_false(-1, 0)), true);
        assert_eq!(r.sample::<bool, _>(&from_ratio(-1, -1)), true);

        assert_eq!(r.sample::<bool, _>(&from_true_false(0, 10)), false);
        assert_eq!(r.sample::<bool, _>(&from_ratio(0, 10)), false);
        assert_eq!(r.sample::<bool, _>(&from_true_false(10, 0)), true);
        assert_eq!(r.sample::<bool, _>(&from_ratio(10, 10)), true);

        assert_eq!(r.sample::<bool, _>(&from_true_false(0, -10)), false);
        assert_eq!(r.sample::<bool, _>(&from_ratio(0, -10)), false);
        assert_eq!(r.sample::<bool, _>(&from_true_false(-10, 0)), true);
        assert_eq!(r.sample::<bool, _>(&from_ratio(-10, -10)), true);
    }

    #[test]
    fn test_trivial() {
        let mut rng = Pcg64::new(0xd30f457389b54ca4a9be12944acfbd14, 0);

        let trues = 5_000_000_000usize;
        let dist = BernoulliExact::from_true_false(trues, 1).unwrap();
        let mut t_count = 0;
        let total = 10_000_000_000usize;
        for _ in 0..total {
            if dist.sample(&mut rng) {
                t_count += 1;
            }
        }
        assert!(
            total - t_count >= 1 && total - t_count <= 2 * (total / trues),
            "t {}\t{:?}",
            t_count,
            dist
        );

        let _dist = BernoulliExact::from_true_false(-5, -10)
            .expect("failed to create BernoulliExact from ratio");
    }

    #[test]
    fn test_matches() {
        let mut rng = Pcg64::new(0xd30f457389b54ca4a9be12944acfbd14, 0);

        for trues in 1..255 {
            'falses: for falses in 1..255 {
                for neg in [false, true] {
                    let dist = if neg {
                        BernoulliExact::from_true_false(-trues, -falses)
                    } else {
                        BernoulliExact::from_true_false(trues, falses)
                    }
                    .expect("failed to create BernoulliExact from ratio");

                    let t_ratio = trues as f64 / (trues + falses) as f64;
                    let f_ratio = falses as f64 / (trues + falses) as f64;

                    let mut t_count = 0;
                    let mut f_count = 0;
                    for _ in 0..10_000 {
                        if dist.sample(&mut rng) {
                            t_count += 1;
                        } else {
                            f_count += 1;
                        }
                    }

                    for _ in 0..1_000_000 {
                        if dist.sample(&mut rng) {
                            t_count += 1;
                        } else {
                            f_count += 1;
                        }
                        let t_count_ratio = t_count as f64 / (t_count + f_count) as f64;
                        let f_count_ratio = f_count as f64 / (t_count + f_count) as f64;

                        let t_err = (t_ratio - t_count_ratio).abs();
                        let f_err = (f_ratio - f_count_ratio).abs();
                        if t_err <= 5e-4 && f_err <= 5e-4 {
                            continue 'falses;
                        }
                    }

                    let t_count_ratio = t_count as f64 / (t_count + f_count) as f64;
                    let f_count_ratio = f_count as f64 / (t_count + f_count) as f64;
                    let t_err = (t_ratio - t_count_ratio).abs();
                    let f_err = (f_ratio - f_count_ratio).abs();
                    assert!(
                        t_err <= 5e-3 && f_err <= 5e-3,
                        "t {} {}\tf {} {}\t{:?}",
                        trues,
                        t_count,
                        falses,
                        f_count,
                        dist
                    );
                }
            }
        }
    }

    #[test]
    fn value_stability() {
        let mut rng = Pcg64::new(0xd30f457389b54ca4a9be12944acfbd14, 0);
        let distr = BernoulliExact::from_ratio(4532, 10000).unwrap();
        let mut buf = [false; 10];
        for x in &mut buf {
            *x = rng.sample(&distr);
        }
        assert_eq!(buf, [
            true, false, true, false, true, false, true, true, true, true
        ]);
    }
}
