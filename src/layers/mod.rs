use crate::tensor::{Shape, Tensor};

pub mod activation;
pub mod convolution;
pub mod deconvolution;
pub mod dense;
pub mod embedding;
pub mod merge;
pub mod pooling;
pub mod softmax;
pub mod split;
// todo: add layers

pub trait Layer {
    const INPUT_DIMENSION: usize;
    type InputType;
    const REVERSE_INPUT_DIMENSION: usize;
    type ReverseInputType;

    const INTERNAL_DIMENSION: usize;
    type InternalType;

    const OUTPUT_DIMENSION: usize;
    type OutputType;
    const REVERSE_OUTPUT_DIMENSION: usize;
    type ReverseOutputType;

    type BPComputation<'s>: FnOnce([Tensor<Self::ReverseOutputType>; Self::REVERSE_OUTPUT_DIMENSION]) ->
    (
        [Tensor<Self::ReverseInputType>; Self::REVERSE_INPUT_DIMENSION],
        [Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]
    ) + 's
        where
            Self: 's,
            [(); Self::INPUT_DIMENSION]:,
            [(); Self::REVERSE_INPUT_DIMENSION]:,
            [(); Self::INTERNAL_DIMENSION]:,
            [(); Self::OUTPUT_DIMENSION]:,
            [(); Self::REVERSE_OUTPUT_DIMENSION]:;

    fn get_input_shapes(&self) -> [Shape; Self::INPUT_DIMENSION];

    fn get_input_shape(&self, n: usize) -> &Shape;

    fn get_output_shapes(&self) -> [Shape; Self::OUTPUT_DIMENSION];

    fn get_output_shape(&self, n: usize) -> &Shape;

    fn feed_forward(
        &self,
        input: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
    ) -> [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION];

    fn back_propagate(
        &self,
        input: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
    ) -> (
        [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION],
        Self::BPComputation<'_>,
    )
        where
            [(); Self::INPUT_DIMENSION]:,
            [(); Self::REVERSE_INPUT_DIMENSION]:,
            [(); Self::INTERNAL_DIMENSION]:,
            [(); Self::OUTPUT_DIMENSION]:,
            [(); Self::REVERSE_OUTPUT_DIMENSION]:;

    fn update(&mut self, update: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]);
}

pub trait LayerBuilder {
    type Layer: Layer;

    fn build(self, input_shapes: [Shape; Self::Layer::INPUT_DIMENSION]) -> Self::Layer; // fixme: Self::Layer::INPUT_DIMENSION in impls not working
}
