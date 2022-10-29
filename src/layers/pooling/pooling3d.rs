use std::marker::PhantomData;

use num_traits::Number;
use void::Void;

use crate::{
    layers::{
        convolution::Padding,
        Layer,
        pooling::{Average, Max},
    },
    tensor::{Shape, Tensor},
};

pub struct Pooling3D<S, T: Number> {
    input_shape: Shape,
    output_shape: Shape,
    pool_size: [usize; 3],
    padding: [(usize, usize); 3],
    strides: [usize; 3],
    dilation: [usize; 3],
    _marker: PhantomData<(S, T)>,
}

impl<S, T: Number> Pooling3D<S, T> {
    pub(crate) fn new(
        input_shape: Shape,
        pool_size: [usize; 3],
        padding: [Padding; 3],
        strides: [usize; 3],
        dilation: [usize; 3],
    ) -> Self {
        let padding = padding.map(Padding::resolve);
        let output_shape = [
            (input_shape[0] + padding[0].0 + padding[0].1 - dilation[0] * (pool_size[0] - 1) - 1) / strides[0] + 1,
            (input_shape[1] + padding[1].0 + padding[1].1 - dilation[1] * (pool_size[1] - 1) - 1) / strides[1] + 1,
            (input_shape[2] + padding[2].0 + padding[2].1 - dilation[2] * (pool_size[2] - 1) - 1) / strides[2] + 1,
            input_shape[3],
        ].into();
        Self {
            input_shape,
            output_shape,
            pool_size,
            padding,
            strides,
            dilation,
            _marker: Default::default(),
        }
    }

    fn iter_through_output<F: FnMut([usize; 4])>(&self, mut f: F) {
        for o0 in 0..self.output_shape[0] {
            for o1 in 0..self.output_shape[1] {
                for o2 in 0..self.output_shape[2] {
                    for o3 in 0..self.output_shape[3] {
                        f([o0, o1, o2, o3])
                    }
                }
            }
        }
    }

    fn iter_through_pool<F: FnMut([usize; 3])>(&self, mut f: F) {
        for p0 in 0..self.pool_size[0] {
            for p1 in 0..self.pool_size[1] {
                for p2 in 0..self.pool_size[2] {
                    f([p0, p1, p2])
                }
            }
        }
    }
}

impl<T: Number + From<i32>> Pooling3D<Average, T> {
    fn feed_forward<F: FnMut([usize; 4], T)>(&self, input: &Tensor<T>, mut f: F) -> Tensor<T> {
        let mut output = Tensor::zero(self.output_shape.clone());
        self.iter_through_output(|[o0, o1, o2, o3]| {
            let mut o = T::zero();
            let mut n = 0;
            self.iter_through_pool(|[p0, p1, p2]| {
                let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                let i2 = (o2 * self.dilation[2] + p2 * self.strides[2]).checked_sub(self.padding[2].0);
                if let (Some(i0), Some(i1), Some(i2)) = (i0, i1, i2) {
                    o += input[[i0, i1, i2, o3]];
                    n += 1;
                }
            });
            let n = T::from(n);
            o /= n;
            f([o0, o1, o2, o3], n);
            output[[o0, o1, o2, o3]] = o
        });
        output
    }
}

impl<T: Number + From<i32>> Layer for Pooling3D<Average, T> {
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
        [Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]
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

    fn feed_forward(&self, [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION]) -> [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION] {
        [self.feed_forward(&input, |_, _| {})]
    }

    fn back_propagate(&self, [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION]) -> ([Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION], Self::BPComputation<'_>) where [(); Self::INPUT_DIMENSION]:, [(); Self::REVERSE_INPUT_DIMENSION]:, [(); Self::INTERNAL_DIMENSION]:, [(); Self::OUTPUT_DIMENSION]:, [(); Self::REVERSE_OUTPUT_DIMENSION]: {
        let mut coverage = Tensor::zero(self.output_shape.clone());
        let output = self.feed_forward(&input, |oi, t| coverage[oi] = t);
        (
            [output],
            move |[output_d]| {
                let mut input_d = Tensor::zero(self.input_shape.clone());
                self.iter_through_output(|[o0, o1, o2, o3]| {
                    let output_d = output_d[[o0, o1, o2, o3]] / coverage[[o0, o1, o2, o3]];
                    self.iter_through_pool(|[p0, p1, p2]| {
                        let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                        let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                        let i2 = (o2 * self.dilation[2] + p2 * self.strides[2]).checked_sub(self.padding[2].0);
                        if let (Some(i0), Some(i1), Some(i2)) = (i0, i1, i2) {
                            input_d[[i0, i1, i2, o3]] += output_d;
                        }
                    });
                });
                ([input_d], [])
            }
        )
    }

    fn update(&mut self, _: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {}
}

impl<T: Number + Ord> Pooling3D<Max, T> {
    fn feed_forward_<F: FnMut([usize; 4], [usize; 4])>(&self, input: &Tensor<T>, mut f: F) -> Tensor<T> {
        let mut output = Tensor::zero(self.output_shape.clone());
        self.iter_through_output(|[o0, o1, o2, o3]| {
            let mut o = T::zero();
            let mut mi = [0, 0, 0, 0];
            self.iter_through_pool(|[p0, p1, p2]| {
                let i0 = (o0 * self.dilation[0] + p0 * self.strides[0]).checked_sub(self.padding[0].0);
                let i1 = (o1 * self.dilation[1] + p1 * self.strides[1]).checked_sub(self.padding[1].0);
                let i2 = (o2 * self.dilation[2] + p2 * self.strides[2]).checked_sub(self.padding[2].0);
                if let (Some(i0), Some(i1), Some(i2)) = (i0, i1, i2) {
                    if input[[i0, i1, i2, o3]] > o {
                        o = input[[i0, i1, i2, o3]];
                        mi = [i0, i1, i2, o3];
                    }
                }
            });
            f([o0, o1, o2, o3], mi);
            output[[o0, o1, o2, o3]] = o
        });
        output
    }
}

impl<T: Number + Ord> Layer for Pooling3D<Max, T> {
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
        [Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]
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

    fn feed_forward(&self, [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION]) -> [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION] {
        [self.feed_forward_(&input, |_, _| {})]
    }

    fn back_propagate(&self, [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION]) -> ([Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION], Self::BPComputation<'_>) where [(); Self::INPUT_DIMENSION]:, [(); Self::REVERSE_INPUT_DIMENSION]:, [(); Self::INTERNAL_DIMENSION]:, [(); Self::OUTPUT_DIMENSION]:, [(); Self::REVERSE_OUTPUT_DIMENSION]: {
        let mut coverage = Tensor::default(self.output_shape.clone());
        let output = self.feed_forward_(&input, |oi, ii| coverage[oi] = ii);
        (
            [output],
            move |[output_d]| {
                let mut input_d = Tensor::zero(self.input_shape.clone());
                self.iter_through_output(|oi| {
                    input_d[coverage[oi]] = output_d[oi];
                });
                ([input_d], [])
            }
        )
    }

    fn update(&mut self, _: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {}
}
