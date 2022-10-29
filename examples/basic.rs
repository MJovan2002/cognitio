#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![feature(type_name_of_val)]
#![allow(incomplete_features)]
#![allow(unused_imports)]

use std::{
    array,
    cmp::Ordering,
    time::{Duration, Instant},
};

use measure::Measurable;

use cognitio::{
    constraints::none::None as NoneConst,
    datasets::{mnist::MNIST, Dataset},
    layers::{
        dense,
        softmax,
        Layer,
        LayerBuilder,
        split,
        merge,
        convolution::{
            self,
            Dim as CDim,
        },
        pooling::{
            self,
            Dim as PDim,
            Max,
        },
        deconvolution::{
            self,
            Dim as DDim,
        },
    },
    losses::{square::Square, Loss},
    model::{
        Model,
        model_tuple::ModelTuple,
    },
    optimizers::{mini_batch::MiniBatch, sgd::SGD},
    regularizers::none::None as NoneReg,
    tensor::{Shape, Tensor},
    trainer::Trainer,
    schedules::{
        polynomial_decay::PolynomialDecay,
        inverse_time_decay::InverseTimeDecay,
    },
    activations::{
        identity::Identity,
        sigmoid::Sigmoid,
        linear::Linear,
    },
};

fn main() {
    #[allow(unused)]
        let new_conv = |a, b, f| convolution::Builder::<{ CDim::Static(2) }, f64, _, _, _, _, _, _, _, _, _, _, _, _, _, _>::new()
        .kernel_shape([a, b])
        .filters(f)
        .activation_default::<Sigmoid<_>>();

    #[allow(unused)]
        let new_pooling = |a, b| pooling::Builder::<{ PDim::Static(2) }, Max, f64, _, _, _, _>::new()
        .pool_shape([a, b]);

    #[allow(unused)]
        let new_deconv = |a, b, f| deconvolution::Builder::<{ DDim::Static(2) }, f64, _, _, _, _, _, _, _, _, _, _, _, _, _>::new()
        .kernel_shape([a, b])
        .filters(f)
        .activation_default::<Identity<_>>();

    #[allow(unused)]
        let new_dense = |output| dense::Builder::new()
        .output_shape([output].into())
        .activation(Linear::new(-1f64, 0f64))
        .kernel_initializer(0f64);

    let mut m = Model::sequential()
        .add_layer(new_dense(16))
        .add_layer(new_dense(10))
        // .add_layer(softmax::Builder::new())
        .build([Shape::new([784])]);

    // println!("{}", std::any::type_name_of_val(&m));

    let mnist = MNIST::offline("examples/training", "examples/testing").unwrap();

    let mut optimizer = SGD::new(0.1);
    let mut trainer = Trainer::new(&mut m, &mut optimizer);
    // println!("{}", std::any::type_name_of_val(&trainer));
    let (t, _) = (|| trainer.train(
        2,
        &mnist,
        |input| [Tensor::from(input.map(|i| i as f64))],
        |[predicted], expected| {
            [Square::new().derive(
                &predicted,
                &Tensor::from(array::from_fn::<_, 10, _>(|i| if i == expected as usize { 1.0 } else { 0.0 })),
            )]
        },
    )).measure();
    println!("{t:?}");

    let (t, s) = (|| mnist.get_training_iter()
        .map(|(input, expected)| ({
                                      let [output] = m.feed_forward([Tensor::from(input.map(|t| t as f64 / 255.0))]);
                                      output.iter()
                                          .copied()
                                          .enumerate()
                                          .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                                          .unwrap()
                                          .0
                                  }, expected as usize))
        .filter(|(output, expected)| output == expected)
        .count()).measure();
    println!("{t:?}");
    println!("{s}/10000 = {}%", s as f64 / 100.0);
}
