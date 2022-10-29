use num_traits::Number;

use crate::{
    optimizers::Optimizer,
    tensor::Tensor,
    schedules::LearningRateSchedule,
};

pub struct SGD<S, G> {
    learning_rate: S,
    momentum: G,
}

impl<S> SGD<S, ()> {
    pub fn new(learning_rate: S) -> Self {
        Self {
            learning_rate,
            momentum: (),
        }
    }
}

impl<S0, S1, G> SGD<S0, (S1, Option<G>)> {
    pub fn momentum(learning_rate: S0, momentum: S1) -> Self {
        Self {
            learning_rate,
            momentum: (momentum, None),
        }
    }
}

impl<S: LearningRateSchedule<f64>, G: Iterable> Optimizer<G> for SGD<S, ()> {
    fn gradients_to_deltas(&mut self, gradients: G) -> Option<G> {
        Some(gradients.modify(Comp {
            alpha: &mut self.learning_rate,
        }))
    }
}

impl<S0: LearningRateSchedule<f64>, S1: LearningRateSchedule<f64>, G: Iterable + Mergeable + Clone> Optimizer<G> for SGD<S0, (S1, Option<G>)> {
    fn gradients_to_deltas(&mut self, gradients: G) -> Option<G> {
        let mut gradients = gradients.modify(Comp {
            alpha: &mut self.learning_rate,
        });

        self.momentum.1 = match self.momentum.1.take() {
            None => Some(gradients.clone()),
            Some(deltas) => {
                let mut deltas = deltas.modify(Comp {
                    alpha: &mut self.momentum.0,
                });
                Mergeable::merge(&mut deltas, &mut gradients);
                Some(deltas)
            }
        };
        Some(gradients)
    }
}

struct Comp<'s, S> {
    alpha: &'s mut S,
}

impl<'s, S: LearningRateSchedule<f64>> Modifier for Comp<'s, S> {
    fn modify<T: Number + From<f64>, const N: usize>(
        &mut self,
        mut input: [Tensor<T>; N],
    ) -> [Tensor<T>; N] {
        let alpha = T::from(self.alpha.next());
        input.iter_mut().for_each(|t| *t *= alpha);
        input
    }
}

trait Iterable {
    fn modify<C: Modifier>(self, c: C) -> Self;
}

impl<A: Number + From<f64>, const N: usize> Iterable for [Tensor<A>; N] {
    fn modify<C: Modifier>(self, mut c: C) -> Self {
        c.modify(self)
    }
}

impl<A: Number + From<f64>, const N: usize, B: Iterable> Iterable for ([Tensor<A>; N], B) {
    fn modify<C: Modifier>(self, mut c: C) -> Self {
        (c.modify(self.0), self.1.modify(c))
    }
}

trait Modifier {
    fn modify<T: Number + From<f64>, const N: usize>(&mut self, input: [Tensor<T>; N]) -> [Tensor<T>; N];
}

trait Mergeable {
    fn merge(a: &mut Self, b: &mut Self);
}

impl<T: Number, const N: usize> Mergeable for [Tensor<T>; N] {
    fn merge(a: &mut Self, b: &mut Self) {
        for i in 0..N {
            a[i].iter_mut().zip(b[i].iter_mut()).for_each(|(a, b)| {
                let t = *a + *b;
                *a = t;
                *b = t;
            })
        }
    }
}

impl<T: Number, const N: usize, B: Mergeable> Mergeable for ([Tensor<T>; N], B) {
    fn merge(a: &mut Self, b: &mut Self) {
        Mergeable::merge(&mut a.0, &mut b.0);
        Mergeable::merge(&mut a.1, &mut b.1);
    }
}
