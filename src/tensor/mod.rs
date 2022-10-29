use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use num_traits::{Float, Number};

pub use shape::Shape;

use crate::tensor::{
    iter::{Iter, IterMut},
    view::{
        view::TensorView,
        view_mut::TensorViewMut,
    },
};

mod iter;
mod shape;
mod view;

#[derive(Clone, Debug)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T> Tensor<T> {
    fn from_parts(data: Vec<T>, shape: Shape) -> Option<Self> {
        if data.len() != shape.capacity() {
            return None;
        }
        Some(Self { data, shape })
    }

    pub fn from<I: into_tensor::IntoTensor<T>>(t: I) -> Self {
        let t = t.into();
        let shape = Shape::new(&t.0);
        Self::from_parts(t.1, shape).unwrap()
    }

    pub fn get_shape(&self) -> &Shape {
        &self.shape
    }

    pub fn iter(&self) -> Iter<T> {
        Iter::new(self.data.iter(), &self.shape)
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self.data.iter_mut(), &self.shape)
    }

    pub fn slice(&self, slices: Vec<SliceIndex>) -> TensorView<T> {
        self.check_slice(&slices);
        TensorView::new(self, slices)
    }

    pub fn slice_mut(&mut self, slices: Vec<SliceIndex>) -> TensorViewMut<T> {
        self.check_slice(&slices);
        TensorViewMut::new(self, slices)
    }

    fn check_slice(&self, slices: &[SliceIndex]) {
        slices.iter().copied().enumerate().for_each(|(i, s)| match s {
            SliceIndex::Range(None, Some(b), _) if b > self.shape[i] => panic!(),
            SliceIndex::Range(Some(a), None, _) if a >= self.shape[i] => panic!(),
            SliceIndex::Range(Some(a), Some(b), _) if a >= b || b > self.shape[i] => panic!(),
            _ => {}
        })
    }
}

impl<T: Number> Tensor<T> {
    pub fn zero<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self {
            data: vec![T::zero(); shape.capacity()],
            shape,
        }
    }

    pub fn clear(&mut self) {
        self.data.iter_mut().for_each(|i| *i = T::zero())
    }
}

impl<T: Default> Tensor<T> {
    pub fn default<S: Into<Shape>>(shape: S) -> Self {
        let shape = shape.into();
        Self {
            data: (0..shape.capacity()).map(|_| T::default()).collect(),
            shape,
        }
    }
}

impl<T: Float> Tensor<T> {
    pub fn is_nan(&self) -> bool {
        self.data.iter().all(|t| !t.is_nan())
    }
}

mod into_tensor {
    pub trait IntoTensor<T> {
        fn into(self) -> (Vec<usize>, Vec<T>);
    }

    pub auto trait NotArray {}

    impl<T, const N: usize> ! NotArray for [T; N] {}
}

impl<T: into_tensor::NotArray> into_tensor::IntoTensor<T> for T {
    fn into(self) -> (Vec<usize>, Vec<T>) {
        (vec![], vec![self])
    }
}

impl<T, U: into_tensor::IntoTensor<T>, const N: usize> into_tensor::IntoTensor<T> for [U; N] {
    fn into(self) -> (Vec<usize>, Vec<T>) {
        let mut iter = self.into_iter();
        let (mut dims, first) = iter.next().unwrap().into();
        dims.push(N);
        let values = first.into_iter().chain(iter.flat_map(|t| t.into().1.into_iter())).collect();
        (dims, values)
    }
}

trait IntoIndex {
    fn into(self, shape: &Shape) -> Option<usize>;
}

impl IntoIndex for usize {
    fn into(self, shape: &Shape) -> Option<usize> {
        if self >= shape.capacity() {
            None
        } else {
            Some(self)
        }
    }
}

trait AsSlice {
    fn as_slice(&self) -> &[usize];
}

impl AsSlice for [usize] {
    fn as_slice(&self) -> &[usize] {
        self
    }
}

impl AsSlice for &[usize] {
    fn as_slice(&self) -> &[usize] {
        *self
    }
}

impl AsSlice for &mut [usize] {
    fn as_slice(&self) -> &[usize] {
        &**self
    }
}

impl<const N: usize> AsSlice for [usize; N] {
    fn as_slice(&self) -> &[usize] {
        self
    }
}

impl<const N: usize> AsSlice for &[usize; N] {
    fn as_slice(&self) -> &[usize] {
        *self
    }
}

impl<const N: usize> AsSlice for &mut [usize; N] {
    fn as_slice(&self) -> &[usize] {
        &**self
    }
}

impl<I: AsSlice> IntoIndex for I {
    fn into(self, shape: &Shape) -> Option<usize> {
        shape.index(self.as_slice())
    }
}

impl<T, I: IntoIndex> Index<I> for Tensor<T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.data[index.into(&self.shape).unwrap()]
    }
}

impl<T, I: IntoIndex> IndexMut<I> for Tensor<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.data[index.into(&self.shape).unwrap()]
    }
}

#[derive(Copy, Clone)]
pub enum SliceIndex {
    Single(usize),
    Range(Option<usize>, Option<usize>, Option<usize>),
}

impl<T: Number> Add<&Self> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: &Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let mut output = self;
        for i in 0..rhs.data.len() {
            output.data[i] += rhs.data[i];
        }
        output
    }
}

impl<T: Number> AddAssign<&Self> for Tensor<T> {
    fn add_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape);
        self.data.iter_mut().zip(rhs.data.iter().cloned()).for_each(|(a, b)| *a += b)
        // for i in 0..self.data.len() {
        //     self.data[i] += rhs.data[i];
        // }
    }
}

impl<T: Number> Sub<&Self> for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: &Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let mut output = self;
        for i in 0..rhs.data.len() {
            output.data[i] -= rhs.data[i];
        }
        output
    }
}

impl<T: Number> SubAssign<&Self> for Tensor<T> {
    fn sub_assign(&mut self, rhs: &Self) {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.data.len() {
            self.data[i] -= rhs.data[i];
        }
    }
}

impl<T: Number> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut output = self.clone();
        for i in 0..self.data.len() {
            output.data[i] = self.data[i] * rhs;
        }
        output
    }
}

impl<T: Number> MulAssign<T> for Tensor<T> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..self.data.len() {
            self.data[i] *= rhs;
        }
    }
}

impl<T: Number> Div<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Self::Output {
        let mut output = self.clone();
        for i in 0..self.data.len() {
            output.data[i] = self.data[i] / rhs;
        }
        output
    }
}

impl<T: Number> DivAssign<T> for Tensor<T> {
    fn div_assign(&mut self, rhs: T) {
        for i in 0..self.data.len() {
            self.data[i] /= rhs;
        }
    }
}

impl<T: Number> Add for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let mut output = self;
        for i in 0..rhs.data.len() {
            output.data[i] += rhs.data[i];
        }
        output
    }
}

impl<T: Number> AddAssign for Tensor<T> {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
    }
}

impl<T: Number> Sub for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);
        let mut output = self;
        for i in 0..rhs.data.len() {
            output.data[i] -= rhs.data[i];
        }
        output
    }
}

impl<T: Number> SubAssign for Tensor<T> {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.shape, rhs.shape);
        for i in 0..self.data.len() {
            self.data[i] -= rhs.data[i];
        }
    }
}

impl<T: Number> Sum for Tensor<T> {
    fn sum<I: Iterator<Item=Self>>(mut iter: I) -> Self {
        let mut output = iter.next().unwrap();
        for t in iter {
            output += t;
        }
        output
    }
}
