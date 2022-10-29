// use num_traits::Number;
//
// use crate::{
//     activations::Activation,
//     initializers::Initializer,
//     layers::{
//         Layer,
//         convolution::Padding
//     },
//     tensor::{Shape, Tensor},
//     constraints::Constraint,
//     regularizers::Regularizer
// };
//
// pub struct Conv<
//     T: Number,
//     A: Activation<T>,
//     KR,
//     BR,
//     AR,
//     KC,
//     BC,
//     const N: usize,
// > {
//     input_shape: Shape,
//     output_shape: Shape,
//     kernel: Tensor<T>,
//     bias: Tensor<T>,
//     activation: A,
//     padding: [(usize, usize); N],
//     strides: [usize; N],
//     dilation: [usize; N],
//     kernel_regularizer: KR,
//     bias_regularizer: BR,
//     activity_regularizer: AR,
//     kernel_constraint: KC,
//     bias_constraint: BC,
// }
//
// impl<
//     T: Number,
//     A: Activation<T>,
//     KR,
//     BR,
//     AR,
//     KC,
//     BC,
//     const N: usize,
// > Conv<T, A, KR, BR, AR, KC, BC, N> where [(); N + 1]: {
//     fn new<KI: Initializer<T>, BI: Initializer<T>>(
//         input_shape: Shape,
//         filters: usize,
//         kernel_shape: [usize; N],
//         mut kernel_initializer: KI,
//         mut bias_initializer: BI,
//         activation: A,
//         padding: [Padding; N],
//         strides: [usize; N],
//         dilation: [usize; N],
//         kernel_regularizer: KR,
//         bias_regularizer: BR,
//         activity_regularizer: AR,
//         kernel_constraint: KC,
//         bias_constraint: BC,
//     ) -> Self {
//         let padding = padding.map(Padding::resolve);
//         let mut output_shape = [0; N + 1];
//         for i in 0..N {
//             output_shape[i] = (input_shape[i] + padding[i].0 + padding[i].1 - dilation[i] * (kernel_shape[i] - 1) - 1) / strides[i] + 1
//         }
//         output_shape[N] = filters;
//         let output_shape = Shape::new(output_shape);
//         Self {
//             kernel: kernel_initializer.initialize(kernel_shape.into()),
//             bias: bias_initializer.initialize(output_shape.clone()),
//             input_shape,
//             output_shape,
//             activation,
//             padding,
//             strides,
//             dilation,
//             kernel_regularizer,
//             bias_regularizer,
//             activity_regularizer,
//             kernel_constraint,
//             bias_constraint,
//         }
//     }
//
//     fn iter_through_output<F: FnMut([usize; N + 1])>(&self, f: F) {
//         let mut index = [0; N + 1];
//         loop {
//             f(index);
//             if !self.output_shape.increment_index(&mut index, 1) {
//                 break;
//             }
//         }
//     }
//
//     fn iter_through_kernel<F: FnMut([usize; N + 2])>(&self, mut f: F) {
//         let mut index = [0; N + 2];
//         loop {
//             f(index);
//             if !self.kernel.get_shape().increment_index(&mut index, self.output_shape[N]) {
//                 break;
//             }
//         }
//     }
//
//     fn feed_forward<F: FnMut([usize; N + 1], T)>(&self, input: &Tensor<T>, mut f: F) -> Tensor<T> {
//         let mut output = Tensor::zero(self.output_shape.clone());
//         self.iter_through_output(|oi| {
//             let mut o = self.bias[oi];
//             self.iter_through_kernel(|ki| {
//                 let mut ii = [0; N + 1];
//                 for i in 0..N {
//                     ii[i] = match (oi[i] * self.dilation[i] + ki[i] * self.strides[i]).checked_sub(self.padding[i].0) {
//                         None => return,
//                         Some(t) => t,
//                     }
//                 }
//                 ii[N] = ki[N];
//                 o += input[ii] * self.kernel[ki]
//             });
//             f(oi, self.activation.derive(o));
//             o = self.activation.activate(o);
//             output[oi] = o
//         });
//         output
//     }
// }
//
// impl<
//     T: Number,
//     A: Activation<T>,
//     KR: Regularizer<T>,
//     BR: Regularizer<T>,
//     AR: Regularizer<T>,
//     KC: Constraint<T>,
//     BC: Constraint<T>,
//     const N: usize,
// > Layer for Conv<T, A, KR, BR, AR, KC, BC, N> where [(); N + 1]: {
//     const INPUT_DIMENSION: usize = 1;
//     type InputType = T;
//     const REVERSE_INPUT_DIMENSION: usize = 1;
//     type ReverseInputType = T;
//
//     const INTERNAL_DIMENSION: usize = 2;
//     type InternalType = T;
//
//     const OUTPUT_DIMENSION: usize = 1;
//     type OutputType = T;
//     const REVERSE_OUTPUT_DIMENSION: usize = 1;
//     type ReverseOutputType = T;
//
//     type BPComputation<'s> = impl FnOnce([Tensor<Self::ReverseOutputType>; Self::REVERSE_OUTPUT_DIMENSION]) ->
//     (
//         [Tensor<Self::ReverseInputType>; Self::REVERSE_INPUT_DIMENSION],
//         [Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION],
//     )
//     where
//     Self: 's,
//     [(); Self::INPUT_DIMENSION]:,
//     [(); Self::REVERSE_INPUT_DIMENSION]:,
//     [(); Self::INTERNAL_DIMENSION]:,
//     [(); Self::OUTPUT_DIMENSION]:,
//     [(); Self::REVERSE_OUTPUT_DIMENSION]: ;
//
//     fn get_input_shapes(&self) -> [Shape; Self::INPUT_DIMENSION] {
//         [self.input_shape.clone()]
//     }
//
//     fn get_input_shape(&self, n: usize) -> &Shape {
//         assert_eq!(n, 0);
//         &self.input_shape
//     }
//
//     fn get_output_shapes(&self) -> [Shape; Self::OUTPUT_DIMENSION] {
//         [self.output_shape.clone()]
//     }
//
//     fn get_output_shape(&self, n: usize) -> &Shape {
//         assert_eq!(n, 0);
//         &self.output_shape
//     }
//
//     fn feed_forward(
//         &self,
//         [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
//     ) -> [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION] {
//         [self.feed_forward(&input, |_, _| {})]
//     }
//
//     fn back_propagate(
//         &self,
//         [input]: [Tensor<Self::InputType>; Self::INPUT_DIMENSION],
//     ) -> (
//         [Tensor<Self::OutputType>; Self::OUTPUT_DIMENSION],
//         Self::BPComputation<'_>,
//     )
//         where
//             [(); Self::INPUT_DIMENSION]:,
//             [(); Self::REVERSE_INPUT_DIMENSION]:,
//             [(); Self::INTERNAL_DIMENSION]:,
//             [(); Self::OUTPUT_DIMENSION]:,
//             [(); Self::REVERSE_OUTPUT_DIMENSION]:,
//     {
//         let mut derivatives = Tensor::zero(self.output_shape.clone());
//         let output = self.feed_forward(&input, |o, t| derivatives[o] = t);
//         let activity_reg = self.activity_regularizer.derive(&output);
//         (
//             [output],
//             move |[output_d]| {
//                 let mut input_d = Tensor::zero(self.input_shape.clone());
//                 let mut kernel_d = self.kernel_regularizer.derive(&self.kernel);
//                 let bias_d = self.bias_regularizer.derive(&self.bias) + &derivatives;
//                 self.iter_through_output(|oi| {
//                     derivatives[oi] *= output_d[oi] + activity_reg[oi];
//                     self.iter_through_kernel(|ki| {
//                         let mut ii = [0; N + 1];
//                         for i in 0..N {
//                             ii[i] = match (oi[i] * self.dilation[i] + ki[i] * self.strides[i]).checked_sub(self.padding[i].0) {
//                                 None => return,
//                                 Some(t) => t,
//                             }
//                         }
//                         ii[N] = ki[N];
//                         kernel_d[ki] += derivatives[oi] * input[ii];
//                         input_d[ii] += derivatives[oi] * kernel_d[ki];
//                     })
//                 });
//                 ([input_d], [kernel_d, bias_d])
//             },
//         )
//     }
//
//     fn update(&mut self, [kernel, bias]: &[Tensor<Self::InternalType>; Self::INTERNAL_DIMENSION]) {
//         self.kernel -= kernel;
//         self.bias -= bias;
//         self.kernel
//             .iter_mut()
//             .for_each(|t| *t = self.kernel_constraint.constrain(*t));
//         self.bias
//             .iter_mut()
//             .for_each(|t| *t = self.bias_constraint.constrain(*t));
//     }
// }
