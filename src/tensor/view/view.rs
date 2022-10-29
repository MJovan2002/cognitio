use std::ops::Index;

use crate::tensor::{
    SliceIndex,
    Tensor,
    view::get_index,
};

pub struct TensorView<'s, T> {
    inner: &'s Tensor<T>,
    slices: Vec<SliceIndex>,
}

impl<'s, T> TensorView<'s, T> {
    pub fn new(tensor: &'s Tensor<T>, slices: Vec<SliceIndex>) -> Self {
        Self {
            inner: tensor,
            slices,
        }
    }
}

impl<T, I: AsRef<[usize]>> Index<I> for TensorView<'_, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.inner[get_index(&self.slices, index).as_slice()]
    }
}