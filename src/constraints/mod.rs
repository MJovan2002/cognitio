pub mod none;
pub mod positive;
// todo: add constraints

pub trait Constraint<T> {
    fn constrain(&self, t: T) -> T;
}
