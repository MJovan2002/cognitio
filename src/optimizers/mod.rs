pub mod mini_batch;
pub mod sgd;

pub trait Optimizer<G> {
    fn gradients_to_deltas(&mut self, gradients: G) -> Option<G>;
}
