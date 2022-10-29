use void::Void;
use crate::layers::Layer;
use crate::tensor::Tensor;

pub trait Model {
    type SubModel;
    type Input;
    type ReverseInput;
    type Internal;
    type Output;
    type ReverseOutput;
    type ReverseType<'s>: FnOnce(Self::ReverseOutput) -> (Self::ReverseInput, Self::Internal) + 's where Self: 's;

    fn feed_forward(&self, input: Self::Input) -> Self::Output;

    fn back_propagate(&self, input: Self::Input) -> (Self::Output, Self::ReverseType<'_>);

    fn update(&mut self, deltas: &Self::Internal);
}

impl<L: Layer + 'static> Model for L where [(); Self::INPUT_DIMENSION]:, [(); Self::REVERSE_INPUT_DIMENSION]:, [(); Self::INTERNAL_DIMENSION]:, [(); Self::OUTPUT_DIMENSION]:, [(); Self::REVERSE_OUTPUT_DIMENSION]: {
    type SubModel = Void;
    type Input = [Tensor<L::InputType>; L::INPUT_DIMENSION];
    type ReverseInput = [Tensor<L::ReverseInputType>; L::REVERSE_INPUT_DIMENSION];
    type Internal = [Tensor<L::InternalType>; L::INTERNAL_DIMENSION];
    type Output = [Tensor<L::OutputType>; L::OUTPUT_DIMENSION];
    type ReverseOutput = [Tensor<L::ReverseOutputType>; L::REVERSE_OUTPUT_DIMENSION];
    type ReverseType<'s> = impl FnOnce(Self::ReverseOutput) -> (Self::ReverseInput, Self::Internal) + 's where Self: 's;

    fn feed_forward(&self, input: Self::Input) -> Self::Output where [(); L::INPUT_DIMENSION]:, [(); L::OUTPUT_DIMENSION]: {
        self.feed_forward(input)
    }

    fn back_propagate(&self, input: Self::Input) -> (Self::Output, Self::ReverseType<'_>) {
        self.back_propagate(input)
    }

    fn update(&mut self, deltas: &Self::Internal) {
        self.update(deltas)
    }
}
