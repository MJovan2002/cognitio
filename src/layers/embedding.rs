use std::marker::PhantomData;

use num_traits::Number;

use crate::{
    layers::{Layer, LayerBuilder},
    tensor::{Shape, Tensor},
    initializers::Initializer,
    constraints::{Constraint, none::None as NoneCon},
    regularizers::{Regularizer, none::None as NoneReg},
};

pub struct Embedding<T, R, C> {
    output_shape: Shape,
    embedding: Tensor<T>,
    regularizer: R,
    constraint: C,
}

impl<T: Number, R, C> Embedding<T, R, C> {
    pub fn new<I: Initializer<T>>(max_size: usize, output_shape: usize, mut initializer: I, regularizer: R, constraint: C) -> Self {
        Self {
            output_shape: [output_shape].into(),
            embedding: initializer.initialize([max_size, output_shape].into()),
            regularizer,
            constraint,
        }
    }
}

impl<T: Number, R: Regularizer<T>, C: Constraint<T>> Layer for Embedding<T, R, C> {
    const INPUT_DIMENSION: usize = 1;
    type InputType = usize;
    const REVERSE_INPUT_DIMENSION: usize = 0;
    type ReverseInputType = ();

    const INTERNAL_DIMENSION: usize = 1;
    type InternalType = T;

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
        [[].into()]
    }

    fn get_input_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        static ZERO_SHAPE: Shape = Shape::zero();
        &ZERO_SHAPE
    }

    fn get_output_shapes(&self) -> [Shape; Self::OUTPUT_DIMENSION] {
        [self.output_shape.clone()]
    }

    fn get_output_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.output_shape
    }

    fn feed_forward(
        &self,
        [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
    ) -> [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION] {
        let index = input[[]];
        let mut output = Tensor::zero(self.output_shape.clone());
        output.iter_mut().enumerate().for_each(|(i, t)| *t = self.embedding[[index, i]]);
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
        let index = input[0][[]];
        let output = self.feed_forward(input);
        let regularization = self.regularizer.derive(&output[0]);
        (
            output,
            move |[output_d]| {
                let l = self.output_shape[1];
                let mut internal_d = Tensor::zero(self.embedding.get_shape().clone());
                for i in 0..l {
                    internal_d[[index, i]] = output_d[i] + regularization[i]
                }
                ([], [internal_d])
            }
        )
    }

    fn update(&mut self, [update]: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {
        self.embedding -= update;
        self.embedding
            .iter_mut()
            .for_each(|t| *t = self.constraint.constrain(*t))
    }
}

pub struct Uninitialized;

auto trait Initialized {}

impl ! Initialized for Uninitialized {}

pub trait IntoInitializer<T> {
    type Initializer: Initializer<T>;

    fn into_initializer(self) -> Self::Initializer;
}

impl<T: Number> IntoInitializer<T> for Uninitialized {
    type Initializer = T;

    fn into_initializer(self) -> Self::Initializer {
        T::zero()
    }
}

impl<T, I: Initializer<T> + Initialized> IntoInitializer<T> for I {
    type Initializer = I;

    fn into_initializer(self) -> Self::Initializer {
        self
    }
}

pub trait IntoRegularizer<T> {
    type Regularizer: Regularizer<T>;

    fn into_regularizer(self) -> Self::Regularizer;
}

impl<T: Number> IntoRegularizer<T> for Uninitialized {
    type Regularizer = NoneReg<T>;

    fn into_regularizer(self) -> Self::Regularizer {
        NoneReg::new()
    }
}

impl<T, R: Regularizer<T> + Initialized> IntoRegularizer<T> for R {
    type Regularizer = R;

    fn into_regularizer(self) -> Self::Regularizer {
        self
    }
}

pub trait IntoConstraint<T> {
    type Constraint: Constraint<T>;

    fn into_constraint(self) -> Self::Constraint;
}

impl<T: Number> IntoConstraint<T> for Uninitialized {
    type Constraint = NoneCon<T>;

    fn into_constraint(self) -> Self::Constraint {
        NoneCon::new()
    }
}

impl<T, C: Constraint<T> + Initialized> IntoConstraint<T> for C {
    type Constraint = C;

    fn into_constraint(self) -> Self::Constraint {
        self
    }
}

pub struct Builder<S, T, I, R, C> {
    output_size: S,
    initializer: I,
    regularizer: R,
    constraint: C,
    _marker: PhantomData<T>,
}

impl<T> Builder<Uninitialized, T, Uninitialized, Uninitialized, Uninitialized> {
    pub fn new() -> Self {
        Self {
            output_size: Uninitialized,
            initializer: Uninitialized,
            regularizer: Uninitialized,
            constraint: Uninitialized,
            _marker: Default::default(),
        }
    }
}

impl<T, I, R, C> Builder<Uninitialized, T, I, R, C> {
    pub fn output_size(self, output_size: usize) -> Builder<usize, T, I, R, C> {
        let Self {
            output_size: _,
            initializer,
            regularizer,
            constraint,
            _marker
        } = self;
        Builder {
            output_size,
            initializer,
            regularizer,
            constraint,
            _marker,
        }
    }
}

impl<S, T, R, C> Builder<S, T, Uninitialized, R, C> {
    pub fn initializer<I: Initializer<T>>(self, initializer: I) -> Builder<S, T, I, R, C> {
        let Self {
            output_size,
            initializer: _,
            regularizer,
            constraint,
            _marker
        } = self;
        Builder {
            output_size,
            initializer,
            regularizer,
            constraint,
            _marker,
        }
    }
}

impl<S, T, I, C> Builder<S, T, I, Uninitialized, C> {
    pub fn regularizer<R: Regularizer<T>>(self, regularizer: R) -> Builder<S, T, I, R, C> {
        let Self {
            output_size,
            initializer,
            regularizer: _,
            constraint,
            _marker
        } = self;
        Builder {
            output_size,
            initializer,
            regularizer,
            constraint,
            _marker,
        }
    }
}

impl<S, T, I, R> Builder<S, T, I, R, Uninitialized> {
    pub fn constraint<C: Constraint<T>>(self, constraint: C) -> Builder<S, T, I, R, C> {
        let Self {
            output_size,
            initializer,
            regularizer,
            constraint: _,
            _marker
        } = self;
        Builder {
            output_size,
            initializer,
            regularizer,
            constraint,
            _marker,
        }
    }
}

impl<
    T: Number,
    I: IntoInitializer<T>,
    R: IntoRegularizer<T>,
    C: IntoConstraint<T>,
> LayerBuilder for Builder<usize, T, I, R, C> {
    type Layer = Embedding<T, R::Regularizer, C::Constraint>;

    fn build(self, [shape]: [Shape; 1]) -> Self::Layer {
        assert_eq!(shape.dimensions(), 1);
        Self::Layer::new(
            shape[0],
            self.output_size,
            self.initializer.into_initializer(),
            self.regularizer.into_regularizer(),
            self.constraint.into_constraint(),
        )
    }
}
