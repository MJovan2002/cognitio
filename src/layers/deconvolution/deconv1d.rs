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

pub struct Deconv1D<
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
    padding: [(usize, usize); 1],
    strides: [usize; 1],
    dilation: [usize; 1],
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
> Deconv1D<T, A, KR, BR, AR, KC, BC> {
    pub(crate) fn new<KI: Initializer<T>, BI: Initializer<T>>(
        input_shape: Shape,
        filters: usize,
        kernel_shape: [usize; 1],
        mut kernel_initializer: KI,
        mut bias_initializer: BI,
        activation: A,
        padding: [Padding; 1],
        strides: [usize; 1],
        dilation: [usize; 1],
        kernel_regularizer: KR,
        bias_regularizer: BR,
        activity_regularizer: AR,
        kernel_constraint: KC,
        bias_constraint: BC,
    ) -> Self {
        assert_eq!(input_shape.dimensions(), 2);
        let padding = padding.map(Padding::resolve);
        let output_shape = Shape::new([
            (input_shape[0] - 1) * strides[0] + dilation[0] * (kernel_shape[0] - 1) + 1 - padding[0].0 - padding[0].1,
            filters,
        ]);
        Self {
            kernel: kernel_initializer.initialize([kernel_shape[0], input_shape[1], filters].into()),
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

    fn iter_through_input<F: FnMut([usize; 2])>(&self, mut f: F) {
        for i0 in 0..self.input_shape[0] {
            for i1 in 0..self.input_shape[1] {
                f([i0, i1])
            }
        }
    }

    fn iter_through_kernel<F: FnMut([usize; 2])>(&self, mut f: F) {
        for k0 in 0..self.kernel.get_shape()[0] {
            for k2 in 0..self.kernel.get_shape()[2] {
                f([k0, k2])
            }
        }
    }

    fn feed_forward<F: FnMut(Vec<usize>, T)>(&self, input: &Tensor<T>, mut f: F) -> Tensor<T> {
        let mut output = self.bias.clone();
        self.iter_through_input(|[i0, i1]| {
            self.iter_through_kernel(|[k0, k2]| {
                let o0 = (i0 * self.dilation[0] + k0 * self.strides[0]).checked_sub(self.padding[0].0);
                if let Some(o0) = o0 {
                    output[[o0, k2]] += input[[i0, i1]] * self.kernel[[k0, i1, k2]]
                }
            });
        });
        output.iter_mut().indexed().for_each(|(oi, t)| {
            f(oi, *t);
            *t = self.activation.activate(*t)
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
> Layer for Deconv1D<T, A, KR, BR, AR, KC, BC> {
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
        let output = self.feed_forward(&input, |o, t| derivatives[o.as_slice()] = t);
        let activation_reg = self.activity_regularizer.derive(&output);
        (
            [output],
            move |[output_d]| {
                let mut input_d = Tensor::zero(self.input_shape.clone());
                let mut kernel_d = self.kernel_regularizer.derive(&self.kernel);
                let bias_d = self.bias_regularizer.derive(&self.bias) + &derivatives;
                derivatives.iter_mut().zip(output_d.iter().copied().zip(activation_reg.iter().copied())).for_each(|(a, (b, c))| *a *= b + c);
                self.iter_through_input(|[i0, i1]| {
                    let mut i = T::zero();
                    self.iter_through_kernel(|[k0, k2]| {
                        let o0 = (i0 * self.dilation[0] + k0 * self.strides[0]).checked_sub(self.padding[0].0);
                        if let Some(o0) = o0 {
                            i += self.kernel[[k0, i1, k2]] * derivatives[[o0, k2]];
                            kernel_d[[k0, i1, k2]] += input[[i0, i1]] * derivatives[[o0, k2]];
                        }
                    });
                    input_d[[i0, i1]] = i;
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
