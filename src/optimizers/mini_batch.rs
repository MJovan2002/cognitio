use std::ops::AddAssign;

use crate::optimizers::Optimizer;

pub struct MiniBatch<O, G> {
    size: usize,
    inner: O,
    position: usize,
    state: Option<G>,
}

impl<O, G> MiniBatch<O, G> {
    pub fn new(size: usize, inner: O) -> Self {
        Self {
            size,
            inner,
            position: 0,
            state: None,
        }
    }
}

impl<O: Optimizer<G>, G: Combinable> Optimizer<G> for MiniBatch<O, G> {
    fn gradients_to_deltas(&mut self, gradients: G) -> Option<G> {
        let deltas = self.inner.gradients_to_deltas(gradients)?;

        self.state = Some(match self.state.take() {
            None => deltas,
            Some(state) => Combinable::combine(state, deltas),
        });

        self.position = (self.position + 1) % self.size;

        if self.position == 0 {
            self.state.take()
        } else {
            None
        }
    }
}

trait Combinable {
    fn combine(a: Self, b: Self) -> Self;
}

impl<T: for<'a> AddAssign<&'a T>, const N: usize> Combinable for [T; N] {
    fn combine(mut a: Self, b: Self) -> Self {
        for i in 0..N {
            a[i] += &b[i];
        }
        a
    }
}

impl<T: for<'a> AddAssign<&'a T>, const N: usize, B: Combinable> Combinable for ([T; N], B) {
    fn combine(a: Self, b: Self) -> Self {
        (Combinable::combine(a.0, b.0), Combinable::combine(a.1, b.1))
    }
}
