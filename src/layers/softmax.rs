use std::cmp::Ordering;
use std::marker::PhantomData;

use num_traits::Float;
use void::Void;

use crate::{
    layers::{Layer, LayerBuilder},
    tensor::{Shape, Tensor},
};

pub struct SoftMax<T> {
    shape: Shape,
    _maker: PhantomData<T>,
}

impl<T> SoftMax<T> {
    fn new(shape: Shape) -> Self {
        Self {
            shape,
            _maker: Default::default(),
        }
    }
}

impl<T: Float> Layer for SoftMax<T> {
    const INPUT_DIMENSION: usize = 1;
    type InputType = T;
    const REVERSE_INPUT_DIMENSION: usize = 1;
    type ReverseInputType = T;

    const INTERNAL_DIMENSION: usize = 0;
    type InternalType = Void;

    const OUTPUT_DIMENSION: usize = 1;
    type OutputType = T;
    const REVERSE_OUTPUT_DIMENSION: usize = 1;
    type ReverseOutputType = T;

    type BPComputation<'s> = impl FnOnce([Tensor<Self::ReverseOutputType>; Self::REVERSE_OUTPUT_DIMENSION]) ->
    (
        [Tensor<Self::ReverseInputType>; Self::REVERSE_INPUT_DIMENSION],
        [Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION],
    ) + 's
    where
    Self: 's,
    [(); Self::INPUT_DIMENSION]:,
    [(); Self::REVERSE_INPUT_DIMENSION]:,
    [(); Self::INTERNAL_DIMENSION]:,
    [(); Self::OUTPUT_DIMENSION]:,
    [(); Self::REVERSE_OUTPUT_DIMENSION]: ;

    fn get_input_shapes(&self) -> [Shape; Self::INPUT_DIMENSION] {
        [self.shape.clone()]
    }

    fn get_input_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.shape
    }

    fn get_output_shapes(&self) -> [Shape; Self::OUTPUT_DIMENSION] {
        [self.shape.clone()]
    }

    fn get_output_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.shape
    }

    fn feed_forward(
        &self,
        [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
    ) -> [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION] {
        let mut output = input;
        let max = output
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or(T::zero());
        output.iter_mut().for_each(|t| *t = (*t - max).exp());
        let s = output.iter().copied().sum::<T>();
        output.iter_mut().for_each(|t| *t /= s);
        [output]
    }

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
            [(); Self::REVERSE_OUTPUT_DIMENSION]:,
    {
        let [output] = self.feed_forward(input);
        (
            [output.clone()],
            move |[mut output_d]| {
                output_d
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, t)| *t *= output[i]);
                let mut input_d = Tensor::zero(output.get_shape().clone());
                input_d
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, t)| *t = output_d.iter().copied().sum::<T>() * *t + output_d[i]);
                ([input_d], [])
            }
        )
    }

    fn update(&mut self, _: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {}
}

pub struct Builder<T> {
    _marker: PhantomData<T>,
}

impl<T> Builder<T> {
    pub fn new() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<T: Float> LayerBuilder for Builder<T> {
    type Layer = SoftMax<T>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(input_shape)
    }
}
