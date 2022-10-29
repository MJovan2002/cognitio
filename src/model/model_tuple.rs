use crate::{
    layers::Layer,
    model::model_trait::Model,
    tensor::Tensor,
};

pub struct ModelTuple<L: Layer + 'static, M: Model + 'static, I0: 'static, O0: 'static, O1: 'static, O2: 'static, I1: 'static, O3: 'static, O4: 'static, O5: 'static> where [(); L::INPUT_DIMENSION]:, [(); L::REVERSE_INPUT_DIMENSION]:, [(); L::INTERNAL_DIMENSION]:, [(); L::OUTPUT_DIMENSION]:, [(); L::REVERSE_OUTPUT_DIMENSION]: {
    pub layer: L,
    pub sub_model: M,
    pub get_input: fn(I0) -> (O0, [Tensor<L::InputType>; L::INPUT_DIMENSION]),
    pub combine_layer_output: fn(O0, [Tensor<L::OutputType>; L::OUTPUT_DIMENSION]) -> (O1, M::Input),
    pub recombine_outputs: fn(O1, M::Output) -> O2,
    pub separate_model_derivatives: fn(O3) -> (O4, M::ReverseOutput),
    pub recombine_derivatives: fn(O4, M::ReverseInput) -> ([Tensor<L::ReverseOutputType>; L::REVERSE_OUTPUT_DIMENSION], O5),
    pub merge_derivatives: fn([Tensor<L::ReverseInputType>; L::REVERSE_INPUT_DIMENSION], O5) -> I1,
}

impl<L: Layer + 'static, M: Model + 'static, I0: 'static, O0: 'static, O1: 'static, O2: 'static, I1: 'static, O3: 'static, O4: 'static, O5: 'static> ModelTuple<L, M, I0, O0, O1, O2, I1, O3, O4, O5> where [(); L::INPUT_DIMENSION]:, [(); L::REVERSE_INPUT_DIMENSION]:, [(); L::INTERNAL_DIMENSION]:, [(); L::OUTPUT_DIMENSION]:, [(); L::REVERSE_OUTPUT_DIMENSION]:, {
    /// ```text
    /// forward:
    ///                |layer_input
    ///                |
    ///                |     |layer_output
    ///                |     |
    ///             |get_input              |recombine_outputs
    ///             |  |     |              |
    ///             V  |     |              V
    ///                |  L  |
    ///             |  V  |  V  |    O1     |
    ///             | --> | --> | --------> |
    ///       I0 -> |     |     |     |     | -> O2
    ///             | --------> | --> | --> |
    ///             |    O0     |  ^  |  ^  |
    ///                            |  M  |
    ///                         ^  |     |
    ///                         |  |     |
    ///                         |combine_layer_output
    ///                            |     |
    ///                            |     |sub_output
    ///                            |
    ///                            |sub_input
    ///
    /// backward:
    ///                |layer_input_derivatives
    ///                |
    ///                |     |layer_output_derivatives
    ///                |     |
    ///             |merge_derivatives      |separate_model_derivatives
    ///             |  |     |              |
    ///             V  |     |              V
    ///                |  L  |
    ///             |  V  |  V  |    O4     |
    ///             | <-- | <-- | <-------- |
    ///       I1 <- |     |     |     |     | <- O3
    ///             | <-------- | <-- | <-- |
    ///             |    O5     |  ^  |  ^  |
    ///                            |  M  |
    ///                         ^  |     |
    ///                         |  |     |
    ///                         |recombine_derivatives
    ///                            |     |
    ///                            |     |sub_output_derivatives
    ///                            |
    ///                            |sub_input_derivatives
    /// ```
    pub const fn new(
        layer: L,
        sub_model: M,
        get_input: fn(I0) -> (O0, [Tensor<L::InputType>; L::INPUT_DIMENSION]),
        combine_layer_output: fn(O0, [Tensor<L::OutputType>; L::OUTPUT_DIMENSION]) -> (O1, M::Input),
        recombine_outputs: fn(O1, M::Output) -> O2,
        separate_model_derivatives: fn(O3) -> (O4, M::ReverseOutput),
        recombine_derivatives: fn(O4, M::ReverseInput) -> ([Tensor<L::ReverseOutputType>; L::REVERSE_OUTPUT_DIMENSION], O5),
        merge_derivatives: fn([Tensor<L::ReverseInputType>; L::REVERSE_INPUT_DIMENSION], O5) -> I1,
    ) -> Self {
        Self {
            layer,
            sub_model,
            get_input,
            combine_layer_output,
            recombine_outputs,
            separate_model_derivatives,
            recombine_derivatives,
            merge_derivatives,
        }
    }
}

impl<L: Layer + 'static, M: Model + 'static, I0: 'static, O0: 'static, O1: 'static, O2: 'static, I1: 'static, O3: 'static, O4: 'static, O5: 'static> Model for ModelTuple<L, M, I0, O0, O1, O2, I1, O3, O4, O5> where [(); L::INPUT_DIMENSION]:, [(); L::REVERSE_INPUT_DIMENSION]:, [(); L::INTERNAL_DIMENSION]:, [(); L::OUTPUT_DIMENSION]:, [(); L::REVERSE_OUTPUT_DIMENSION]: {
    type SubModel = M;
    type Input = I0;
    type ReverseInput = I1;
    type Internal = (
        [Tensor<L::InternalType>; L::INTERNAL_DIMENSION],
        M::Internal,
    );
    type Output = O2;
    type ReverseOutput = O3;
    type ReverseType<'s> = impl FnOnce(Self::ReverseOutput) -> (Self::ReverseInput, Self::Internal) + 's where Self: 's;

    fn feed_forward(&self, input: Self::Input) -> Self::Output {
        let (remaining_input, layer_input) = (self.get_input)(input);
        let layer_output = self.layer.feed_forward(layer_input);
        let (remaining_output, sub_input) = (self.combine_layer_output)(remaining_input, layer_output);
        let sub_output = self.sub_model.feed_forward(sub_input);
        (self.recombine_outputs)(remaining_output, sub_output)
    }

    fn back_propagate(&self, input: Self::Input) -> (Self::Output, Self::ReverseType<'_>) {
        let (remaining_input, layer_input) = (self.get_input)(input);
        let (layer_output, layer_computation) = self.layer.back_propagate(layer_input);
        let (remaining_output, sub_input) = (self.combine_layer_output)(remaining_input, layer_output);
        let (sub_output, sub_computation) = self.sub_model.back_propagate(sub_input);
        let output = (self.recombine_outputs)(remaining_output, sub_output);
        (
            output,
            |derivatives| {
                let (remaining_output_derivatives, sub_output_derivatives) = (self.separate_model_derivatives)(derivatives);
                let (sub_input_derivatives, sub_internal) = sub_computation(sub_output_derivatives);
                let (layer_output_derivatives, remaining_output_derivatives) = (self.recombine_derivatives)(remaining_output_derivatives, sub_input_derivatives);
                let (layer_input_derivatives, layer_internal) = layer_computation(layer_output_derivatives);
                let input_derivatives = (self.merge_derivatives)(layer_input_derivatives, remaining_output_derivatives);
                (input_derivatives, (layer_internal, sub_internal))
            },
        )
    }

    fn update(&mut self, deltas: &Self::Internal) {
        self.layer.update(&deltas.0);
        self.sub_model.update(&deltas.1);
    }
}
