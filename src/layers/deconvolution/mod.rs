use std::marker::PhantomData;
use num_traits::Number;

use crate::{
    constraints::{Constraint, none::None as NoneCon},
    initializers::Initializer,
    regularizers::{Regularizer, none::None as NoneReg},
    activations::Activation,
    layers::{
        deconvolution::{
            deconv1d::Deconv1D,
            deconv2d::Deconv2D,
            deconv3d::Deconv3D,
        },
        LayerBuilder,
        convolution::Padding,
    },
    tensor::Shape,
};

mod deconv;
mod deconv1d;
mod deconv2d;
mod deconv3d;

pub struct Uninitialized;

auto trait Initialized {}

impl ! Initialized for Uninitialized {}

trait IntoPadding<const N: usize> {
    fn into_padding(self) -> [Padding; N];
}

impl<const N: usize> IntoPadding<N> for Uninitialized {
    fn into_padding(self) -> [Padding; N] {
        [Padding::None; N]
    }
}

impl<const N: usize> IntoPadding<N> for [Padding; N] {
    fn into_padding(self) -> [Padding; N] {
        self
    }
}

trait IntoStride<const N: usize> {
    fn into_stride(self) -> [usize; N];
}

impl<const N: usize> IntoStride<N> for Uninitialized {
    fn into_stride(self) -> [usize; N] {
        [1; N]
    }
}

impl<const N: usize> IntoStride<N> for [usize; N] {
    fn into_stride(self) -> [usize; N] {
        self
    }
}

trait IntoDilation<const N: usize> {
    fn into_dilation(self) -> [usize; N];
}

impl<const N: usize> IntoDilation<N> for Uninitialized {
    fn into_dilation(self) -> [usize; N] {
        [1; N]
    }
}

impl<const N: usize> IntoDilation<N> for [usize; N] {
    fn into_dilation(self) -> [usize; N] {
        self
    }
}

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

#[derive(Eq, PartialEq, Copy, Clone)]
pub enum Dim {
    Static(usize),
    Dynamic(usize),
}

impl Dim {
    pub const fn dim(self) -> usize {
        match self {
            Dim::Static(t) => t,
            Dim::Dynamic(t) => t,
        }
    }
}

pub struct Builder<const N: Dim, T: Number, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
    kernel_shape: SHAPE,
    filters: F,
    activation: A,
    padding: P,
    strides: S,
    dilation: D,
    kernel_initializer: KI,
    bias_initializer: BI,
    kernel_regularizer: KR,
    bias_regularizer: BR,
    activity_regularizer: AR,
    kernel_constraint: KC,
    bias_constraint: BC,
    _marker: PhantomData<T>,
}

impl<const N: Dim, T: Number> Builder<N, T, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized, Uninitialized> {
    pub fn new() -> Self {
        Self {
            kernel_shape: Uninitialized,
            filters: Uninitialized,
            activation: Uninitialized,
            padding: Uninitialized,
            strides: Uninitialized,
            dilation: Uninitialized,
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

impl<const N: Dim, T: Number, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> Builder<N, T, Uninitialized, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
    pub fn kernel_shape(
        self,
        kernel_shape: [usize; N.dim()],
    ) -> Builder<N, T, [usize; N.dim()], F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape: _,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> Builder<N, T, SHAPE, Uninitialized, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
    pub fn filters(
        self,
        filters: usize,
    ) -> Builder<N, T, SHAPE, usize, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters: _,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, P, S, D, KI, BI, KR, BR, AR, KC, BC> Builder<N, T, SHAPE, F, Uninitialized, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
    pub fn activation<A: Activation<T>>(
        self,
        activation: A,
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation: _,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        self.activation(A::default())
    }
}

impl<const N: Dim, T: Number, SHAPE, F, A, S, D, KI, BI, KR, BR, AR, KC, BC> Builder<N, T, SHAPE, F, A, Uninitialized, S, D, KI, BI, KR, BR, AR, KC, BC> {
    pub fn padding(
        self,
        padding: [Padding; N.dim()],
    ) -> Builder<N, T, SHAPE, F, A, [Padding; N.dim()], S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding: _,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, D, KI, BI, KR, BR, AR, KC, BC> Builder<N, T, SHAPE, F, A, P, Uninitialized, D, KI, BI, KR, BR, AR, KC, BC> {
    pub fn strides(
        self,
        strides: [usize; N.dim()],
    ) -> Builder<N, T, SHAPE, F, A, P, [usize; N.dim()], D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides: _,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, S, KI, BI, KR, BR, AR, KC, BC> Builder<N, T, SHAPE, F, A, P, S, Uninitialized, KI, BI, KR, BR, AR, KC, BC> {
    pub fn dilation(
        self,
        dilation: [usize; N.dim()],
    ) -> Builder<N, T, SHAPE, F, A, P, S, [usize; N.dim()], KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation: _,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, S, D, BI, KR, BR, AR, KC, BC> Builder<N, T, SHAPE, F, A, P, S, D, Uninitialized, BI, KR, BR, AR, KC, BC> {
    pub fn kernel_initializer<KI: Initializer<T>>(
        self,
        kernel_initializer: KI,
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, S, D, KI, KR, BR, AR, KC, BC> Builder<N, T, SHAPE, F, A, P, S, D, KI, Uninitialized, KR, BR, AR, KC, BC> {
    pub fn bias_initializer<BI: Initializer<T>>(
        self,
        bias_initializer: BI,
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, S, D, KI, BI, BR, AR, KC, BC> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, Uninitialized, BR, AR, KC, BC> {
    pub fn kernel_regularizer<KR: Regularizer<T>>(
        self,
        kernel_regularizer: KR,
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, S, D, KI, BI, KR, AR, KC, BC> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, Uninitialized, AR, KC, BC> {
    pub fn bias_regularizer<BR: Regularizer<T>>(
        self,
        bias_regularizer: BR,
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, S, D, KI, BI, KR, BR, KC, BC> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, Uninitialized, KC, BC> {
    pub fn activity_regularizer<AR: Regularizer<T>>(
        self,
        activity_regularizer: AR,
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, BC> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, Uninitialized, BC> {
    pub fn kernel_constraint<KC: Constraint<T>>(
        self,
        kernel_constraint: KC,
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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

impl<const N: Dim, T: Number, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, Uninitialized> {
    pub fn bias_constraint<BC: Constraint<T>>(
        self,
        bias_constraint: BC,
    ) -> Builder<N, T, SHAPE, F, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
        let Self {
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
            kernel_shape,
            filters,
            activation,
            padding,
            strides,
            dilation,
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
    P: IntoPadding<1>,
    S: IntoStride<1>,
    D: IntoDilation<1>,
    KI: IntoInitializer<T>,
    BI: IntoInitializer<T>,
    KR: IntoRegularizer<T>,
    BR: IntoRegularizer<T>,
    AR: IntoRegularizer<T>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
> LayerBuilder for Builder<{ Dim::Static(1) }, T, [usize; 1], usize, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
    type Layer = Deconv1D<T, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.filters,
            self.kernel_shape,
            self.kernel_initializer.into_initializer(),
            self.bias_initializer.into_initializer(),
            self.activation,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}

impl<
    T: Number,
    A: Activation<T>,
    P: IntoPadding<2>,
    S: IntoStride<2>,
    D: IntoDilation<2>,
    KI: IntoInitializer<T>,
    BI: IntoInitializer<T>,
    KR: IntoRegularizer<T>,
    BR: IntoRegularizer<T>,
    AR: IntoRegularizer<T>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
> LayerBuilder for Builder<{ Dim::Static(2) }, T, [usize; 2], usize, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
    type Layer = Deconv2D<T, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.filters,
            self.kernel_shape,
            self.kernel_initializer.into_initializer(),
            self.bias_initializer.into_initializer(),
            self.activation,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}

impl<
    T: Number,
    A: Activation<T>,
    P: IntoPadding<3>,
    S: IntoStride<3>,
    D: IntoDilation<3>,
    KI: IntoInitializer<T>,
    BI: IntoInitializer<T>,
    KR: IntoRegularizer<T>,
    BR: IntoRegularizer<T>,
    AR: IntoRegularizer<T>,
    KC: IntoConstraint<T>,
    BC: IntoConstraint<T>,
> LayerBuilder for Builder<{ Dim::Static(3) }, T, [usize; 3], usize, A, P, S, D, KI, BI, KR, BR, AR, KC, BC> {
    type Layer = Deconv3D<T, A, KR::Regularizer, BR::Regularizer, AR::Regularizer, KC::Constraint, BC::Constraint>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.filters,
            self.kernel_shape,
            self.kernel_initializer.into_initializer(),
            self.bias_initializer.into_initializer(),
            self.activation,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
            self.kernel_regularizer.into_regularizer(),
            self.bias_regularizer.into_regularizer(),
            self.activity_regularizer.into_regularizer(),
            self.kernel_constraint.into_constraint(),
            self.bias_constraint.into_constraint(),
        )
    }
}
