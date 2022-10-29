use std::marker::PhantomData;

use num_traits::Float;

use crate::activations::Activation;

pub struct Identity<T> {
    _marker: PhantomData<T>,
}

impl<T> Identity<T> {
    pub fn new() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<T> Default for Identity<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float + From<i32>> Activation<T> for Identity<T> {
    fn activate(&self, x: T) -> T {
        x
    }

    fn derive(&self, _: T) -> T {
        T::from(1)
    }
}
