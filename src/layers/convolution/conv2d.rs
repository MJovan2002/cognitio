use num_traits::Number;

use crate::{
    activations::Activation,
    initializers::Initializer,
    layers::{
        Layer,
        convolution::Padding,
    },
    tensor::{Shape, Tensor},
    constraints::Constraint,
    regularizers::Regularizer,
};

pub struct Conv2D<
    T: Number,
    A: Activation<T>,
    KR,
    BR,
    AR,
    KC,
    BC,
> {
    input_shape: Shape,
    output_shape: Shape,
    kernel: Tensor<T>,
    bias: Tensor<T>,
    activation: A,
    padding: [(usize, usize); 2],
    strides: [usize; 2],
    dilation: [usize; 2],
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
> Conv2D<T, A, KR, BR, AR, KC, BC> {
    pub(crate) fn new<KI: Initializer<T>, BI: Initializer<T>>(
        input_shape: Shape,
        filters: usize,
        kernel_shape: [usize; 2],
        mut kernel_initializer: KI,
        mut bias_initializer: BI,
        activation: A,
        padding: [Padding; 2],
        strides: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    ) -> Self {
        assert_eq!(input_shape.dimensions(), 3);
        assert_eq!(filters % groups, 0);
        assert_eq!(input_shape[2] % groups, 0);
        let padding = padding.map(Padding::resolve);
        let output_shape = Shape::new([
            (input_shape[0] + padding[0].0 + padding[0].1 - dilation[0] * (kernel_shape[0] - 1) - 1) / strides[0] + 1,
            (input_shape[1] + padding[1].0 + padding[1].1 - dilation[1] * (kernel_shape[1] - 1) - 1) / strides[1] + 1,
            filters,
        ]);
        Self {
            kernel: kernel_initializer.initialize([kernel_shape[0], kernel_shape[1], input_shape[2] / groups, filters].into()),
            bias: bias_initializer.initialize(output_shape.clone()),
            input_shape,
            output_shape,
            activation,
            padding,
            strides,
            dilation,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
        }
    }

    fn iter_through_output<F: FnMut([usize; 3])>(&self, mut f: F) {
        for o0 in 0..self.output_shape[0] {
            for o1 in 0..self.output_shape[1] {
                for o2 in 0..self.output_shape[2] {
                    f([o0, o1, o2])
                }
            }
        }
    }

    fn iter_through_kernel<F: FnMut([usize; 3])>(&self, mut f: F) {
        for k0 in 0..self.kernel.get_shape()[0] {
            for k1 in 0..self.kernel.get_shape()[1] {
                for k2 in 0..self.kernel.get_shape()[2] {
                    f([k0, k1, k2])
                }
            }
        }
    }

    fn feed_forward<F: FnMut([usize; 3], T)>(&self, input: &Tensor<T>, mut f: F) -> Tensor<T> {
        let mut output = Tensor::zero(self.output_shape.clone());
        self.iter_through_output(|[o0, o1, o2]| {
            let mut o = self.bias[[o0, o1, o2]];
            self.iter_through_kernel(|[k0, k1, k2]| {
                let i0 = (o0 * self.dilation[0] + k0 * self.strides[0]).checked_sub(self.padding[0].0);
                let i1 = (o1 * self.dilation[1] + k1 * self.strides[1]).checked_sub(self.padding[1].0);
                if let (Some(i0), Some(i1)) = (i0, i1) {
                    let groups = self.input_shape[2] / self.kernel.get_shape()[2];
                    for g in (0..groups).map(|g| g * self.kernel.get_shape()[2]) {
                        o += input[[i0, i1, g + k2]] * self.kernel[[k0, k1, k2, o2]]
                    }
                }
            });
            f([o0, o1, o2], self.activation.derive(o));
            o = self.activation.activate(o);
            output[[o0, o1, o2]] = o
        });
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
> Layer for Conv2D<T, A, KR, BR, AR, KC, BC> {
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
    )
        where
            [(); Self::INPUT_DIMENSION]:,
            [(); Self::REVERSE_INPUT_DIMENSION]:,
            [(); Self::INTERNAL_DIMENSION]:,
            [(); Self::OUTPUT_DIMENSION]:,
            [(); Self::REVERSE_OUTPUT_DIMENSION]:,
    {
        let mut derivatives = Tensor::zero(self.output_shape.clone());
        let output = self.feed_forward(&input, |o, t| derivatives[o] = t);
        let activation_reg = self.activity_regularizer.derive(&output);
        (
            [output],
            move |[output_d]| {
                let mut input_d = Tensor::zero(self.input_shape.clone());
                let mut kernel_d = self.kernel_regularizer.derive(&self.kernel);
                let bias_d = self.bias_regularizer.derive(&self.bias) + &derivatives;
                self.iter_through_output(|[o0, o1, o2]| {
                    derivatives[[o0, o1, o2]] *= output_d[[o0, o1, o2]] + activation_reg[[o0, o1, o2]];
                    self.iter_through_kernel(|[k0, k1, k2]| {
                        let i0 = (o0 * self.dilation[0] + k0 * self.strides[0]).checked_sub(self.padding[0].0);
                        let i1 = (o1 * self.dilation[1] + k1 * self.strides[1]).checked_sub(self.padding[1].0);
                        if let (Some(i0), Some(i1)) = (i0, i1) {
                            let groups = self.input_shape[2] / self.kernel.get_shape()[2];
                            for g in (0..groups).map(|g| g * self.kernel.get_shape()[2]) {
                                kernel_d[[k0, k1, k2, o2]] += input[[i0, i1, g + k2]] * derivatives[[o0, o1, o2]];
                                input_d[[i0, i1, g + k2]] += kernel_d[[k0, k1, k2, o2]] * derivatives[[o0, o1, o2]];
                            }
                        }
                    })
                });
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
