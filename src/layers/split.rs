use std::array;
use std::marker::PhantomData;

use num_traits::Number;
use void::Void;

use crate::{
    layers::{Layer, LayerBuilder},
    tensor::{Shape, Tensor},
};

pub struct Split<T, const N: usize> {
    shape: Shape,
    _maker: PhantomData<T>,
}

impl<T, const N: usize> Split<T, N> {
    fn new(shape: Shape) -> Self {
        Self {
            shape,
            _maker: Default::default(),
        }
    }
}

impl<T: Number + Clone, const N: usize> Layer for Split<T, N> {
    const INPUT_DIMENSION: usize = 1;
    type InputType = T;
    const REVERSE_INPUT_DIMENSION: usize = 1;
    type ReverseInputType = T;

    const INTERNAL_DIMENSION: usize = 0;
    type InternalType = Void;

    const OUTPUT_DIMENSION: usize = N;
    type OutputType = T;
    const REVERSE_OUTPUT_DIMENSION: usize = N;
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
        array::from_fn(|_| self.shape.clone())
    }

    fn get_output_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.shape
    }

    fn feed_forward(
        &self,
        [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
    ) -> [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION] {
        array::from_fn(|_| input.clone())
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
        (
            self.feed_forward(input),
            |output_d| {
                let mut i = Tensor::zero(output_d[0].get_shape().clone());
                for t in output_d {
                    i += &t;
                }
                ([i], [])
            }
        )
    }

    fn update(&mut self, _: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {}
}

pub struct Builder<T, const N: usize> {
    _marker: PhantomData<T>,
}

impl<T, const N: usize> Builder<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: Default::default(),
        }
    }
}

impl<T: Number, const N: usize> LayerBuilder for Builder<T, N> {
    type Layer = Split<T, N>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(input_shape)
    }
}
