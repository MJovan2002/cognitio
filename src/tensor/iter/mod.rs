use std::marker::PhantomData;

pub use iter::Iter;
pub use iter_mut::IterMut;

use crate::tensor::Shape;

mod iter;
mod iter_mut;

pub trait TensorIterator<'s>: Iterator {
    fn position(&self) -> usize;

    fn get_shape(&self) -> &Shape;
}

pub struct Indexed<'s, I: TensorIterator<'s>> {
    iter: I,
    index: Vec<usize>,
    _marker: PhantomData<&'s ()>,
}

impl<'s, I: TensorIterator<'s>> Indexed<'s, I> {
    pub(crate) fn new(iter: I) -> Self {
        let index = iter.get_shape().inverse_index(iter.position());
        Self {
            iter,
            index,
            _marker: Default::default(),
        }
    }
}

impl<'s, I: TensorIterator<'s>> Iterator for Indexed<'s, I> {
    type Item = (Vec<usize>, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let t = self.iter.next()?;
        let t = Some((self.index.clone(), t));
        self.iter.get_shape().increment_index(&mut self.index, 1);
        t
    }
}
