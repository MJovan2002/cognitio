use num_traits::Number;

use crate::{regularizers::Regularizer, tensor::Tensor};

pub struct L2<T: Number> {
    alpha: T,
}

impl<T: Number + From<i32>> L2<T> {
    pub fn new(alpha: T) -> Self {
        Self { alpha }
    }

    pub fn one() -> Self {
        Self {
            alpha: T::from(1),
        }
    }
}

impl<T: Number + From<i32>> Regularizer<T> for L2<T> {
    fn regularization(&self, tensor: &Tensor<T>) -> T {
        tensor.iter().map(|&t| t * t).sum::<T>() * self.alpha
    }

    fn derive(&self, tensor: &Tensor<T>) -> Tensor<T> {
        let mut t = Tensor::zero(tensor.get_shape().clone());
        t.iter_mut()
            .zip(tensor.iter())
            .for_each(|(a, &b)| *a = T::from(2) * b * self.alpha);
        t
    }
}
