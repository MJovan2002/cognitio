use num_traits::Number;

use crate::tensor::Tensor;

pub mod square;

pub trait Loss<T: Number> {
    fn loss(&self, predicted: &Tensor<T>, expected: &Tensor<T>) -> T;

    fn derive(&self, predicted: &Tensor<T>, expected: &Tensor<T>) -> Tensor<T>;
}
