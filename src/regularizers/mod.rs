use crate::tensor::Tensor;

pub mod l1;
pub mod l2;
pub mod none;

pub trait Regularizer<T> {
    fn regularization(&self, tensor: &Tensor<T>) -> T;

    fn derive(&self, tensor: &Tensor<T>) -> Tensor<T>;
}
