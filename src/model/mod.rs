use sequential::Sequential;
use crate::model::sequential::Empty;

mod sequential;
pub mod model_tuple;
pub(crate) mod model_trait;

pub struct Model<M> {
    pub model: M,
}

impl<M: model_trait::Model> Model<M> {
    pub(crate) fn from_inner(model: M) -> Self {
        Self { model }
    }

    // pub fn new(inputs: Inputs, outputs: Outputs) -> Self {} todo

    pub fn feed_forward(&self, input: M::Input) -> M::Output {
        self.model.feed_forward(input)
    }

    pub fn back_propagate(&self, input: M::Input) -> (M::Output, M::ReverseType<'_>) {
        self.model.back_propagate(input)
    }

    pub fn update(&mut self, deltas: &M::Internal) {
        self.model.update(deltas)
    }
}

impl Model<Empty> {
    pub const fn sequential() -> Sequential<Empty> {
        Sequential::empty()
    }
}
