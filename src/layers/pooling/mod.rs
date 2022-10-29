use std::marker::PhantomData;

use num_traits::Number;

use crate::{
    layers::{
        convolution::Padding,
        LayerBuilder,
        pooling::{
            pooling1d::Pooling1D,
            pooling2d::Pooling2D,
            pooling3d::Pooling3D,
        },
    },
    tensor::Shape,
};

pub mod pooling;
pub mod pooling1d;
pub mod pooling2d;
pub mod pooling3d;

pub struct Average;

pub struct Max;

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

pub struct Builder<const N: Dim, M, T: Number, SHAPE, P, S, D> {
    pool_shape: SHAPE,
    padding: P,
    strides: S,
    dilation: D,
    _marker: PhantomData<(M, T)>,
}

impl<const N: Dim, M, T: Number> Builder<N, M, T, Uninitialized, Uninitialized, Uninitialized, Uninitialized> {
    pub fn new() -> Self {
        Self {
            pool_shape: Uninitialized,
            padding: Uninitialized,
            strides: Uninitialized,
            dilation: Uninitialized,
            _marker: Default::default(),
        }
    }
}

impl<const N: Dim, M, T: Number, P, S, D> Builder<N, M, T, Uninitialized, P, S, D> {
    pub fn pool_shape(
        self,
        pool_shape: [usize; N.dim()],
    ) -> Builder<N, M, T, [usize; N.dim()], P, S, D> {
        let Self {
            pool_shape: _,
            padding,
            strides,
            dilation,
            _marker,
        } = self;
        Builder {
            pool_shape,
            padding,
            strides,
            dilation,
            _marker,
        }
    }
}

impl<const N: Dim, M, T: Number, SHAPE, S, D> Builder<N, M, T, SHAPE, Uninitialized, S, D> {
    pub fn padding(
        self,
        padding: [Padding; N.dim()],
    ) -> Builder<N, M, T, SHAPE, [Padding; N.dim()], S, D> {
        let Self {
            pool_shape,
            padding: _,
            strides,
            dilation,
            _marker,
        } = self;
        Builder {
            pool_shape,
            padding,
            strides,
            dilation,
            _marker,
        }
    }
}

impl<const N: Dim, M, T: Number, SHAPE, P, D> Builder<N, M, T, SHAPE, P, Uninitialized, D> {
    pub fn strides(
        self,
        strides: [usize; N.dim()],
    ) -> Builder<N, M, T, SHAPE, P, [usize; N.dim()], D> {
        let Self {
            pool_shape,
            padding,
            strides: _,
            dilation,
            _marker,
        } = self;
        Builder {
            pool_shape,
            padding,
            strides,
            dilation,
            _marker,
        }
    }
}

impl<const N: Dim, M, T: Number, SHAPE, P, S> Builder<N, M, T, SHAPE, P, S, Uninitialized> {
    pub fn dilation(
        self,
        dilation: [usize; N.dim()],
    ) -> Builder<N, M, T, SHAPE, P, S, [usize; N.dim()]> {
        let Self {
            pool_shape,
            padding,
            strides,
            dilation: _,
            _marker,
        } = self;
        Builder {
            pool_shape,
            padding,
            strides,
            dilation,
            _marker,
        }
    }
}

impl<
    T: Number + From<i32>,
    P: IntoPadding<1>,
    S: IntoStride<1>,
    D: IntoDilation<1>,
> LayerBuilder for Builder<{ Dim::Static(1) }, Average, T, [usize; 1], P, S, D> {
    type Layer = Pooling1D<Average, T>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.pool_shape,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
        )
    }
}

impl<
    T: Number + Ord,
    P: IntoPadding<1>,
    S: IntoStride<1>,
    D: IntoDilation<1>,
> LayerBuilder for Builder<{ Dim::Static(1) }, Max, T, [usize; 1], P, S, D> {
    type Layer = Pooling1D<Max, T>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.pool_shape,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
        )
    }
}

impl<
    T: Number + From<i32>,
    P: IntoPadding<2>,
    S: IntoStride<2>,
    D: IntoDilation<2>,
> LayerBuilder for Builder<{ Dim::Static(2) }, Average, T, [usize; 2], P, S, D> {
    type Layer = Pooling2D<Average, T>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.pool_shape,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
        )
    }
}

impl<
    T: Number + Ord,
    P: IntoPadding<2>,
    S: IntoStride<2>,
    D: IntoDilation<2>,
> LayerBuilder for Builder<{ Dim::Static(2) }, Max, T, [usize; 2], P, S, D> {
    type Layer = Pooling2D<Max, T>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.pool_shape,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
        )
    }
}

impl<
    T: Number + From<i32>,
    P: IntoPadding<3>,
    S: IntoStride<3>,
    D: IntoDilation<3>,
> LayerBuilder for Builder<{ Dim::Static(3) }, Average, T, [usize; 3], P, S, D> {
    type Layer = Pooling3D<Average, T>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.pool_shape,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
        )
    }
}

impl<
    T: Number + Ord,
    P: IntoPadding<3>,
    S: IntoStride<3>,
    D: IntoDilation<3>,
> LayerBuilder for Builder<{ Dim::Static(3) }, Max, T, [usize; 3], P, S, D> {
    type Layer = Pooling3D<Max, T>;

    fn build(self, [input_shape]: [Shape; 1]) -> Self::Layer {
        Self::Layer::new(
            input_shape,
            self.pool_shape,
            self.padding.into_padding(),
            self.strides.into_stride(),
            self.dilation.into_dilation(),
        )
    }
}
