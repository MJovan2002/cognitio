use num_traits::Float;

use crate::activations::Activation;

pub struct ELU<T: Float> {
    alpha: T,
}

impl<T: Float> ELU<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }
}

impl<T: Float + From<i32>> Activation<T> for ELU<T> {
    fn activate(&self, x: T) -> T {
        if x > T::zero() {
            x
        } else {
            self.alpha * (x.exp() - T::from(1))
        }
    }

    fn derive(&self, x: T) -> T {
        if x > T::zero() {
            T::from(1)
        } else {
            self.alpha * x.exp()
        }
    }
}
