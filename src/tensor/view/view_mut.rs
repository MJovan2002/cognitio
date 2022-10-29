use std::ops::{Index, IndexMut};

use crate::tensor::{
    SliceIndex,
    Tensor,
    view::get_index,
};

pub struct TensorViewMut<'s, T> {
    inner: &'s mut Tensor<T>,
    slices: Vec<SliceIndex>,
}

impl<'s, T> TensorViewMut<'s, T> {
    pub fn new(tensor: &'s mut Tensor<T>, slices: Vec<SliceIndex>) -> Self {
        Self {
            inner: tensor,
            slices,
        }
    }
}

impl<T, I: AsRef<[usize]>> Index<I> for TensorViewMut<'_, T> {
    type Output = T;

    fn index(&self, index: I) -> &Self::Output {
        &self.inner[get_index(&self.slices, index).as_slice()]
    }
}

impl<T, I: AsRef<[usize]>> IndexMut<I> for TensorViewMut<'_, T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.inner[get_index(&self.slices, index).as_slice()]
    }
}