#![feature(auto_traits)]
#![feature(negative_impls)]
#![feature(generic_const_exprs)]
#![feature(type_alias_impl_trait)]
#![feature(const_trait_impl)]
#![feature(adt_const_params)]
#![feature(allocator_api)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(associated_type_bounds)]
// #![feature(associated_const_equality)]
// #![feature(array_zip)]
// #![feature(generic_arg_infer)]
// #![feature(split_array)]
// #![feature(const_convert)]
// #![feature(iter_advance_by)]

#![allow(incomplete_features)]
// #![forbid(unsafe_code)]
// #![deny(missing_docs)]

//! # Cognito
//! ## Blazingly fast and modular ML framework
//!
//! Basic building block of the framework is [`Tensor`]-s and [`Layer`]-s.
//! [`Tensor`]-s are used to pass data around.
//! [`Layer`] is a trait implemented by structs that modify data.
//! [`Layer`] takes an array of [`Tensor`]-s and returns another.
//! It can also return derivatives of some [`Loss`] functions relative to all it's internal states and inputs it received.
//!
//! [`Layer`]-s are organised into [`Model`].
//!
//! There are two types of [`Model`]-s:
//! * `Sequential`
//! ```
//! # use std::array;
//! # use cognitio::activations::sigmoid::Sigmoid;
//! # use cognitio::model::Model;
//! # use cognitio::layers::dense;
//! # use cognitio::datasets::mnist::MNIST;
//! # use cognitio::optimizers::sgd::SGD;
//! # use cognitio::losses::square::Square;
//! #
//! let model = Model::sequential()
//!     .add_layer(dense::Builder::new()
//!         .output_shape([32].into())
//!         .activation_default::<Sigmoid<_>>())
//!     .add_layer(dense::Builder::new()
//!         .output_shape([10].into())
//!         .activation_default::<Sigmoid<_>>())
//!     .build([[784].into()]);
//! #
//! # let dataset = MNIST::offline("examples/training", "examples/testing")
//! #     .unwrap();
//! #
//! # let optimizer = Sgd::new(0.1);
//! #
//! # let trainer = Trainer::new(&mut model, &mut optimizer);
//! # trainer.train(
//! #     1,
//! #     &dataset,
//! #     |input| [Tensor::from(input.map(From::from))],
//! #     |[predicted], expected| [Square::new().derive(
//! #         &predicted,
//! #         &Tensor::from(array::from_fn::<_, 10, _>(
//! #             |i| (i == expected as usize) as _)
//! #         )
//! #     )]
//! # )
//! ```
//! * `Graph` TODO
//!
//! Once you created a [`Model`] you can construct a [`Dataset`].
//! ```
//! # use std::array;
//! # use cognitio::activations::sigmoid::Sigmoid;
//! # use cognitio::model::Model;
//! # use cognitio::layers::dense;
//! # use cognitio::datasets::mnist::MNIST;
//! # use cognitio::optimizers::sgd::SGD;
//! # use cognitio::losses::square::Square;
//! #
//! # let model = Model::sequential()
//! #     .add_layer(dense::Builder::new()
//! #         .output_shape([32].into())
//! #         .activation_default::<Sigmoid<_>>())
//! #     .add_layer(dense::Builder::new()
//! #         .output_shape([10].into())
//! #         .activation_default::<Sigmoid<_>>())
//! #     .build([[784].into()]);
//! #
//! let dataset = MNIST::offline("examples/training", "examples/testing")
//!     .unwrap();
//! #
//! # let optimizer = Sgd::new(0.1);
//! #
//! # let trainer = Trainer::new(&mut model, &mut optimizer);
//! # trainer.train(
//! #     1,
//! #     &dataset,
//! #     |input| [Tensor::from(input.map(From::from))],
//! #     |[predicted], expected| [Square::new().derive(
//! #         &predicted,
//! #         &Tensor::from(array::from_fn::<_, 10, _>(
//! #             |i| (i == expected as usize) as _)
//! #         )
//! #     )]
//! # )
//! ```
//! After that choose an [`Optimizer`].
//! ```
//! # use std::array;
//! # use cognitio::activations::sigmoid::Sigmoid;
//! # use cognitio::model::Model;
//! # use cognitio::layers::dense;
//! # use cognitio::datasets::mnist::MNIST;
//! # use cognitio::optimizers::sgd::SGD;
//! # use cognitio::losses::square::Square;
//! #
//! # let model = Model::sequential()
//! #     .add_layer(dense::Builder::new()
//! #         .output_shape([32].into())
//! #         .activation_default::<Sigmoid<_>>())
//! #     .add_layer(dense::Builder::new()
//! #         .output_shape([10].into())
//! #         .activation_default::<Sigmoid<_>>())
//! #     .build([[784].into()]);
//! #
//! # let dataset = MNIST::offline("examples/training", "examples/testing")
//! #     .unwrap();
//! #
//! let optimizer = Sgd::new(0.1);
//! #
//! # let trainer = Trainer::new(&mut model, &mut optimizer);
//! # trainer.train(
//! #     1,
//! #     &dataset,
//! #     |input| [Tensor::from(input.map(From::from))],
//! #     |[predicted], expected| [Square::new().derive(
//! #         &predicted,
//! #         &Tensor::from(array::from_fn::<_, 10, _>(
//! #             |i| (i == expected as usize) as _)
//! #         )
//! #     )]
//! # )
//! ```
//! Finally, create a [`Trainer`] using the [`Model`] and the [`Optimizer`] and train the model on the [`Dataset`].
//! ```
//! # use std::array;
//! # use cognitio::activations::sigmoid::Sigmoid;
//! # use cognitio::model::Model;
//! # use cognitio::layers::dense;
//! # use cognitio::datasets::mnist::MNIST;
//! # use cognitio::optimizers::sgd::SGD;
//! # use cognitio::losses::square::Square;
//! #
//! # let model = Model::sequential()
//! #     .add_layer(dense::Builder::new()
//! #         .output_shape([32].into())
//! #         .activation_default::<Sigmoid<_>>())
//! #     .add_layer(dense::Builder::new()
//! #         .output_shape([10].into())
//! #         .activation_default::<Sigmoid<_>>())
//! #     .build([[784].into()]);
//! #
//! # let dataset = MNIST::offline("examples/training", "examples/testing")
//! #     .unwrap();
//! #
//! # let optimizer = Sgd::new(0.1);
//! #
//! let trainer = Trainer::new(&mut model, &mut optimizer);
//! trainer.train(
//!     1,
//!     &dataset,
//!     |input| [Tensor::from(input.map(From::from))],
//!     |[predicted], expected| [Square::new().derive(
//!         &predicted,
//!         &Tensor::from(array::from_fn::<_, 10, _>(
//!             |i| (i == expected as usize) as _)
//!         )
//!     )]
//! )
//! ```
//! [`Tensor`]: self::tensor::Tensor
//! [`Layer`]: self::layers::Layer
//! [`Loss`]: self::losses::Loss
//! [`Model`]: self::model::Model
//! [`Trainer`]: self::trainer::Trainer
//! [`Dataset`]: self::datasets::Dataset
//! [`Optimizer`]: self::optimizers::Optimizer

pub mod activations;
pub mod callbacks;
pub mod constraints;
pub mod datasets;
pub mod initializers;
pub mod layers;
pub mod losses;
pub mod metrics;
pub mod model;
pub mod optimizers;
pub mod regularizers;
pub mod schedules;
pub mod tensor;
pub mod trainer;

// pub mod prelude{
//     pub use crate::{
//         activations::{
//             elu::ELU,
//             exp::EXP,
//             sigmoid::Sigmoid,
//             identity::Identity,
//             linear::Linear,
//             relu::ReLU,
//             softplus::SoftPlus,
//             softsign::SoftSign,
//             swish::Swish,
//             tanh::Tanh,
//         },
//         constraints::{
//             none::None as NoneConst,
//             positive::Positive,
//         },
//         datasets::mnist::MNIST,
//         initializers::constant,
//         layers::{
//             *,
//             convolution::*,
//         },
//         model::{
//             Model,
//             model_tuple::ModelTuple, // todo: remove
//         },
//         optimizers::{
//             mini_batch::MiniBatch,
//             sgd::SGD,
//         },
//         regularizers::{
//             none::None as NoneReg,
//             l1::L1,
//             l2::L2
//         },
//         schedules::{
//             exponential_decay::ExponentialDecay,
//             polynomial_decay::PolynomialDecay,
//             inverse_time_decay::InverseTimeDecay,
//         },
//         tensor::{
//             Tensor,
//             Shape,
//         },
//         trainer::Trainer,
//     };
// }

// todo: add prelude
// todo: encapsulate all model outputs
// todo: add losses to model
// todo: enable no_std
