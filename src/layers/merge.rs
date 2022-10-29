use std::{
    array,
    marker::PhantomData,
};

use num_traits::Number;
use void::Void;

#[allow(unused)]
use crate::{
    layers::{Layer, LayerBuilder},
    tensor::{Shape, Tensor},
};

pub struct Merge<T, const N: usize> {
    shape: Shape,
    _marker: PhantomData<T>,
}

impl<T, const N: usize> Merge<T, N> {
    pub fn new(shape: Shape) -> Self {
        Self { shape, _marker: Default::default() }
    }
}

impl<T: Number, const N: usize> Layer for Merge<T, N> {
    const INPUT_DIMENSION: usize = N;
    type InputType = T;
    const REVERSE_INPUT_DIMENSION: usize = N;
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
    ) + 's where
    Self: 's,
    [(); Self::INPUT_DIMENSION]:,
    [(); Self::REVERSE_INPUT_DIMENSION]:,
    [(); Self::INTERNAL_DIMENSION]:,
    [(); Self::OUTPUT_DIMENSION]:,
    [(); Self::REVERSE_OUTPUT_DIMENSION]: ;

    fn get_input_shapes(&self) -> [Shape; Self::INPUT_DIMENSION] {
        array::from_fn(|_| self.shape.clone())
    }

    fn get_input_shape(&self, n: usize) -> &Shape {
        assert!(n < N);
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
        input: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
    ) -> [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION] {
        [input.into_iter().sum()]
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
            |[output_d]| (array::from_fn(|_| output_d.clone()), []), // todo: remove custom combining function
        )
    }

    fn update(&mut self, _: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {}
}

pub struct Builder<T, F, const N: usize> {
    #[allow(unused)]
    f: F,
    _marker: PhantomData<T>,
}

impl<T, const N: usize> Builder<T, (), N> {
    pub fn add_merge_function(self, f: fn([T; N]) -> T) -> Builder<T, fn([T; N]) -> T, N> {
        Builder {
            f,
            _marker: Default::default(),
        }
    }
}

// fixme: const generics error

// impl<T: Number, const N: usize> LayerBuilder for Builder<T, fn([T; N]) -> T, N> {
//     type Layer = Merge<T, N>;
//
//     fn build(self, input_shapes: [Shape; Self::Layer::INPUT_DIMENSION]) -> Self::Layer {
//         Self::Layer::new(input_shapes[0].clone(), self.f)
//     }
// }
