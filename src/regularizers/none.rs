use std::marker::PhantomData;

use num_traits::Number;

use crate::{regularizers::Regularizer, tensor::Tensor};

pub struct None<T> {
    _marker: PhantomData<T>,
}

impl<T> None<T> {
    pub fn new() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<T: Number> Regularizer<T> for None<T> {
    #[inline]
    fn regularization(&self, _: &Tensor<T>) -> T {
        T::zero()
    }

    fn derive(&self, tensor: &Tensor<T>) -> Tensor<T> {
        Tensor::zero(tensor.get_shape().clone())
    }
}
