use num_traits::Number;

use crate::{
    initializers::Initializer,
    tensor::{Shape, Tensor},
};

impl<T: Number> Initializer<T> for T {
    fn initialize(&mut self, shape: Shape) -> Tensor<T> {
        let mut t = Tensor::zero(shape);
        t.iter_mut().for_each(|i| *i = *self);
        t
    }
}
