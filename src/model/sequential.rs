#![allow(unused_qualifications)]

use std::marker::Destruct;

use crate::{
    layers::{Layer, LayerBuilder},
    model::{model_trait, Model, model_tuple::ModelTuple},
    tensor::{Shape, Tensor},
};

pub struct Empty;

pub struct Sequential<M> {
    inner: M,
}

impl<M> Sequential<M> {
    const fn new(inner: M) -> Self {
        Self { inner }
    }
}

impl Sequential<Empty> {
    pub(crate) const fn empty() -> Self {
        Self::new(Empty)
    }
}

impl<M: Insertable> Sequential<M> {
    pub fn add_layer<L: LayerBuilder>(self, layer: L) -> Sequential<M::Output<L>> {
        Sequential::new(self.inner.insert(layer))
    }
}

impl<M: Buildable> Sequential<M> where M::Output: model_trait::Model {
    pub fn build(self, input: M::Input) -> Model<M::Output> {
        Model::from_inner(self.inner.build(input))
    }
}

#[derive(Debug)]
pub struct Pair<A, B>(A, B);

pub auto trait Singular {}

impl<A, B> ! Singular for Pair<A, B> {}

impl ! Singular for Empty {}

#[const_trait]
pub trait Insertable {
    type Output<T>;

    fn insert<T>(self, t: T) -> Self::Output<T>;
}

impl const Insertable for Empty {
    type Output<T> = T;

    fn insert<T>(self, t: T) -> Self::Output<T> {
        t
    }
}

impl<A: Singular> const Insertable for A {
    type Output<T> = Pair<A, T>;

    fn insert<T>(self, t: T) -> Self::Output<T> {
        Pair(self, t)
    }
}

impl<A: ~ const Destruct, B: ~ const Insertable + ~ const Destruct> const Insertable for Pair<A, B> {
    type Output<T> = Pair<A, B::Output<T>>;

    fn insert<T>(self, t: T) -> Self::Output<T> {
        let Self(a, b) = self;
        Pair(a, b.insert(t))
    }
}

pub trait Buildable {
    type Input;
    type Output;

    fn build(self, input: Self::Input) -> Self::Output;
}

impl<B: LayerBuilder> Buildable for B where [(); B::Layer::INPUT_DIMENSION]:, [(); B::Layer::OUTPUT_DIMENSION]:, [(); B::Layer::REVERSE_INPUT_DIMENSION]:, [(); B::Layer::REVERSE_OUTPUT_DIMENSION]: {
    type Input = [Shape; B::Layer::INPUT_DIMENSION];
    type Output = B::Layer;

    fn build(self, input: Self::Input) -> Self::Output {
        self.build(input)
    }
}

impl<B: LayerBuilder<Layer: 'static>, M: Buildable<Input=[Shape; B::Layer::OUTPUT_DIMENSION]>> Buildable for Pair<B, M>
    where M::Output: model_trait::Model<
        Input=[Tensor<<B::Layer as Layer>::OutputType>; B::Layer::OUTPUT_DIMENSION],
        ReverseInput=[Tensor<<B::Layer as Layer>::ReverseOutputType>; B::Layer::REVERSE_OUTPUT_DIMENSION]
    > + 'static, [(); B::Layer::INPUT_DIMENSION]:, [(); B::Layer::REVERSE_INPUT_DIMENSION]:, [(); B::Layer::INTERNAL_DIMENSION]:, [(); B::Layer::OUTPUT_DIMENSION]:, [(); B::Layer::REVERSE_OUTPUT_DIMENSION]: {
    type Input = <B as Buildable>::Input;
    type Output = ModelTuple<
        B::Layer,
        M::Output,
        [Tensor<<B::Layer as Layer>::InputType>; B::Layer::INPUT_DIMENSION],
        (),
        (),
        <M::Output as model_trait::Model>::Output,
        [Tensor<<B::Layer as Layer>::ReverseInputType>; B::Layer::REVERSE_INPUT_DIMENSION],
        <M::Output as model_trait::Model>::ReverseOutput,
        (),
        ()
    >;

    fn build(self, input: Self::Input) -> Self::Output {
        let layer = self.0.build(input);
        let sub_model = self.1.build(layer.get_output_shapes());
        ModelTuple::new(
            layer,
            sub_model,
            |input| ((), input),
            |_, layer_output| ((), layer_output),
            |_, model_output| model_output,
            |model_derivatives| ((), model_derivatives),
            |_, layer_derivatives| (layer_derivatives, ()),
            |input_derivatives, _| input_derivatives,
        )
    }
}
