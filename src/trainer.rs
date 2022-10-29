use crate::{
    datasets::Dataset,
    model::{model_trait, Model},
    optimizers::Optimizer,
};

pub struct Trainer<'m, 'o, M: model_trait::Model, O: Optimizer<M::Internal>> {
    model: &'m mut Model<M>,
    optimizer: &'o mut O,
}

impl<'m, 'o, M: model_trait::Model, O: Optimizer<M::Internal>> Trainer<'m, 'o, M, O> {
    pub fn new(model: &'m mut Model<M>, optimizer: &'o mut O) -> Self {
        Self { model, optimizer }
    }

    pub fn train<DS: Dataset>(
        &mut self,
        epochs: usize,
        dataset: &DS,
        preprocess: fn(DS::Input) -> M::Input,
        loss: fn(M::Output, DS::Label) -> M::ReverseOutput,
    ) { // todo: add metrics and callbacks, integrate losses into model
        for _ in 0..epochs {
            for (input, expected) in dataset.get_training_iter().map(|(a, b)| (preprocess(a), b)) {
                let (predicted, computation) = self.model.back_propagate(input);
                let derivatives = loss(predicted, expected);
                let (_, gradients) = computation(derivatives);
                if let Some(deltas) = self.optimizer.gradients_to_deltas(gradients) {
                    self.model.update(&deltas);
                }
            }
        }
    }
}
