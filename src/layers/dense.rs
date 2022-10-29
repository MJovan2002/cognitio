use std::marker::PhantomData;

use num_traits::Number;

use crate::{
    activations::Activation,
    constraints::{Constraint, none::None as NoneCon},
    initializers::Initializer,
    layers::{Layer, LayerBuilder},
    regularizers::{Regularizer, none::None as NoneReg},
    tensor::{Shape, Tensor},
};

pub struct Dense<
    T: Number,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
> {
    kernel: Tensor<T>,
    bias: Tensor<T>,
    input_shape: Shape,
    output_shape: Shape,
    activation: A,
    kernel_regularizer: KR,
    bias_regularizer: BR,
    activity_regularizer: AR,
    kernel_constraint: KC,
    bias_constraint: BC,
}

impl<
    T: Number,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
> Dense<T, A, KR, BR, AR, KC, BC>
{
    fn new<KI: Initializer<T>, BI: Initializer<T>>(
        input_shape: Shape,
        output_shape: Shape,
        activation: A,
        mut kernel_initializer: KI,
        mut bias_initializer: BI,
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    ) -> Self {
        Self {
            kernel: kernel_initializer.initialize([
                input_shape.capacity(),
                output_shape.capacity(),
            ].into()),
            bias: bias_initializer.initialize(output_shape.clone()),
            input_shape,
            output_shape,
            activation,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
        }
    }

    fn feed_forward<F: FnMut(usize, T)>(&self, input: &Tensor<T>, mut f: F) -> Tensor<T> {
        let mut output = Tensor::zero(self.bias.get_shape().clone());
        for o in 0..self.kernel.get_shape()[1] {
            let mut out = self.bias[o];
            for i in 0..self.kernel.get_shape()[0] {
                out += input[i] * self.kernel[[i, o]];
            }
            output[o] = self.activation.activate(out);
            f(o, out);
        }
        output
    }
}

impl<
    T: Number,
    A: Activation<T>,
    KR: Regularizer<T>,
    BR: Regularizer<T>,
    AR: Regularizer<T>,
    KC: Constraint<T>,
    BC: Constraint<T>,
> Layer for Dense<T, A, KR, BR, AR, KC, BC>
{
    const INPUT_DIMENSION: usize = 1;
    type InputType = T;
    const REVERSE_INPUT_DIMENSION: usize = 1;
    type ReverseInputType = T;

    const INTERNAL_DIMENSION: usize = 2;
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
        [self.input_shape.clone()]
    }

    fn get_input_shape(&self, n: usize) -> &Shape {
        assert_eq!(n, 0);
        &self.input_shape
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
        [self.feed_forward(&input, |_, _| {})]
    }

    fn back_propagate(
        &self,
        [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
    ) -> (
        [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION],
        Self::BPComputation<'_>,
    ) where [(); Self::INPUT_DIMENSION]:, [(); Self::REVERSE_INPUT_DIMENSION]:, [(); Self::INTERNAL_DIMENSION]:, [(); Self::OUTPUT_DIMENSION]:, [(); Self::REVERSE_OUTPUT_DIMENSION]:, {
        let mut derivatives = Tensor::zero(self.output_shape.clone());
        let output = self.feed_forward(&input, |i, t| derivatives[i] = self.activation.derive(t));
        let activation_reg = self.activity_regularizer.derive(&output);
        (
            [output],
            move |[output_d]| {
                derivatives
                    .iter_mut()
                    .zip(output_d.iter().zip(activation_reg.iter()))
                    .for_each(|(d, (od, ar))| *d *= *od + *ar);
                let mut input_d = Tensor::zero(self.input_shape.clone());
                let mut kernel_d = self.kernel_regularizer.derive(&self.kernel);
                let bias_d = self.bias_regularizer.derive(&self.bias) + &derivatives;
                for i in 0..self.kernel.get_shape()[0] {
                    let mut id = T::zero();
                    for o in 0..self.kernel.get_shape()[1] {
                        kernel_d[[i, o]] = derivatives[o] * input[i];
                        id += derivatives[o] * self.kernel[[i, o]];
                    }
                    input_d[i] = id;
                }
                ([input_d], [kernel_d, bias_d])
            },
        )
    }

    fn update(&mut self, [kernel, bias]: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {
        self.kernel -= kernel;
        self.bias -= bias;
        self.kernel
            .iter_mut()
            .for_each(|t| *t = self.kernel_constraint.constrain(*t));
        self.bias
            .iter_mut()
            .for_each(|t| *t = self.bias_constraint.constrain(*t));
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

pub struct Builder<T: Number, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
    output_shape: SHAPE,
    activation: A,
    kernel_initializer: KI,
    bias_initializer: BI,
    kernel_regularizer: KR,
    bias_regularizer: BR,
    activity_regularizer: AR,
    kernel_constraint: KC,
    bias_constraint: BC,
    _marker: PhantomData<T>,
}

impl<T: Number> Builder<T, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized> {
    pub fn new() -> Self {
        Self {
            output_shape: Uninitialized,
            activation: Uninitialized,
            kernel_initializer: Uninitialized,
            bias_initializer: Uninitialized,
            kernel_regularizer: Uninitialized,
            bias_regularizer: Uninitialized,
            activity_regularizer: Uninitialized,
            kernel_constraint: Uninitialized,
            bias_constraint: Uninitialized,
            _marker: Default::default(),
        }
    }
}

impl<T: Number, A, KI, BI, KR, BR, AR, KC, BC> Builder<T, Uninitialized, A, KI, BI, KR, BR, AR, KC, BC> {
    pub fn output_shape(
        self,
        output_shape: Shape,
    ) -> Builder<T, Shape, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape: _,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }
}

impl<T: Number, SHAPE, KI, BI, KR, BR, AR, KC, BC> Builder<T, SHAPE, Uninitialized, KI, BI, KR, BR, AR, KC, BC> {
    pub fn activation<A: Activation<T>>(
        self,
        activation: A,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape,
            activation: _,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }

    pub fn activation_default<A: Activation<T> + Default>(
        self,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        self.activation(A::default())
    }
}

impl<T: Number, SHAPE, A, BI, KR, BR, AR, KC, BC> Builder<T, SHAPE, A, Uninitialized, BI, KR, BR, AR, KC, BC> {
    pub fn kernel_initializer<KI: Initializer<T>>(
        self,
        kernel_initializer: KI,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape,
            activation,
            kernel_initializer: _,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }
}

impl<T: Number, SHAPE, A, KI, KR, BR, AR, KC, BC> Builder<T, SHAPE, A, KI, Uninitialized, KR, BR, AR, KC, BC> {
    pub fn bias_initializer<BI: Initializer<T>>(
        self,
        bias_initializer: BI,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer: _,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }
}

impl<T: Number, SHAPE, A, KI, BI, BR, AR, KC, BC> Builder<T, SHAPE, A, KI, BI, Uninitialized, BR, AR, KC, BC> {
    pub fn kernel_regularizer<KR: Regularizer<T>>(
        self,
        kernel_regularizer: KR,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer: _,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }
}

impl<T: Number, SHAPE, A, KI, BI, KR, AR, KC, BC> Builder<T, SHAPE, A, KI, BI, KR, Uninitialized, AR, KC, BC> {
    pub fn bias_regularizer<BR: Regularizer<T>>(
        self,
        bias_regularizer: BR,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer: _,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }
}

impl<T: Number, SHAPE, A, KI, BI, KR, BR, KC, BC> Builder<T, SHAPE, A, KI, BI, KR, BR, Uninitialized, KC, BC> {
    pub fn activity_regularizer<AR: Regularizer<T>>(
        self,
        activity_regularizer: AR,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer: _,
            kernel_constraint,
            bias_constraint,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }
}

impl<T: Number, SHAPE, A, KI, BI, KR, BR, AR, BC> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, Uninitialized, BC> {
    pub fn kernel_constraint<KC: Constraint<T>>(
        self,
        kernel_constraint: KC,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint: _,
            bias_constraint,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }
}

impl<T: Number, SHAPE, A, KI, BI, KR, BR, AR, KC> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, Uninitialized> {
    pub fn bias_constraint<BC: Constraint<T>>(
        self,
        bias_constraint: BC,
    ) -> Builder<T, SHAPE, A, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint: _,
            _marker,
        } = self;
        Builder {
            output_shape,
            activation,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            _marker,
        }
    }
}

impl<
    T: Number,
    A: Activation<T>,
    KI: IntoInitializer<T>,
    BI: IntoInitializer<T>,
    KR: IntoRegularizer<T>,
    BR: IntoRegularizer<T>,
    AR: IntoRegularizer<T>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
> LayerBuilder for Builder<T, Shape, A, KI, BI, KR, BR, AR, KC, BC> {
    type Layer = Dense<T, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.output_shape,
            self.activation,
            self.kernel_initializer.into_initializer(),
            self.bias_initializer.into_initializer(),
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}
