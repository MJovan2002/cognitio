use crate::tensor::{Shape, Tensor};

pub mod constant;
// todo: add initializers

pub trait Initializer<T> {
    fn initialize(&mut self, shape: Shape) -> Tensor<T>;
}
