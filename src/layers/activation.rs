use std::marker::PhantomData;

use num_traits::Number;
use void::Void;

use crate::{
    activations::Activation as Act,
    layers::{Layer, LayerBuilder},
    tensor::{Shape, Tensor},
};

pub struct Activation<T: Number, A: Act<T>> {
    activation: A,
    shape: Shape,
    _marker: PhantomData<T>,
}

impl<T: Number, A: Act<T>> Activation<T, A> {
    pub fn new(activation: A, shape: Shape) -> Self {
        Self {
            activation,
            shape,
            _marker: Default::default(),
        }
    }

    fn feed_forward<F: FnMut(usize, T)>(&self, input: &Tensor<T>, mut f: F) -> [Tensor<T>; 1] {
        let mut o = input.clone();
        o.iter_mut().enumerate().for_each(|(i, t)| {
            *t = self.activation.activate(*t);
            f(i, *t);
        });
        [o]
    }
}

impl<T: Number, A: Act<T>> Layer for Activation<T, A> {
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
    where Self: 's,
    [(); Self::INPUT_DIMENSION]:,
    [(); Self::INTERNAL_DIMENSION]:,
    [(); Self::OUTPUT_DIMENSION]: ;

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
        self.feed_forward(&input, |_, _| {})
    }

    fn back_propagate(
        &self,
        [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
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
        let mut derivatives = Tensor::zero(self.shape.clone());
        (
            self.feed_forward(&input, |i, t| derivatives[i] = t),
            |[output_d]| {
                derivatives
                    .iter_mut()
                    .zip(output_d.iter())
                    .for_each(|(a, b)| *a *= *b);
                ([derivatives], [])
            },
        )
    }

    fn update(&mut self, _: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {}
}

pub struct Builder<T: Number, A> {
    activation: A,
    _marker: PhantomData<T>,
}

impl<T: Number> Builder<T, ()> {
    pub fn new() -> Self {
        Self {
            activation: (),
            _marker: Default::default(),
        }
    }

    pub fn activation<A: Act<T>>(self, activation: A) -> Builder<T, A> {
        Builder {
            activation,
            _marker: Default::default(),
        }
    }
}

impl<T: Number, A: Act<T>> LayerBuilder for Builder<T, A> {
    type Layer = Activation<T, A>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(self.activation, input_shape)
    }
}
