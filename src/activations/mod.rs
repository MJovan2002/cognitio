pub mod elu;
pub mod exp;
pub mod identity;
pub mod linear;
pub mod relu;
pub mod sigmoid;
pub mod softplus;
pub mod softsign;
pub mod swish;
pub mod tanh;

pub trait Activation<T> {
    fn activate(&self, x: T) -> T;

    fn derive(&self, x: T) -> T;
}
