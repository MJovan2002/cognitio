use std::marker::PhantomData;

use num_traits::Number;

use crate::{losses::Loss, tensor::Tensor};

pub struct Square<T: Number> {
    _marker: PhantomData<T>,
}

impl<T: Number> Square<T> {
    pub fn new() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<T: Number + From<i32>> Loss<T> for Square<T> {
    fn loss(&self, predicted: &Tensor<T>, expected: &Tensor<T>) -> T {
        predicted
            .iter()
            .zip(expected.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum()
    }

    fn derive(&self, predicted: &Tensor<T>, expected: &Tensor<T>) -> Tensor<T> {
        let mut out = Tensor::zero(predicted.get_shape().clone());
        for i in 0..predicted.get_shape().capacity() {
            out[i] = T::from(2) * (predicted[i] - expected[i])
        }
        out
    }
}
