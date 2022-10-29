use std::{
    cmp::Ordering,
    iter::{Product, Sum},
    marker::PhantomData,
};

use crate::tensor::{
    iter::Indexed,
    iter::TensorIterator,
    Shape,
};

pub struct Iter<'s, T> {
    iter: std::slice::Iter<'s, T>,
    position: usize,
    shape: &'s Shape,
    _marker: PhantomData<T>,
}

impl<'s, T> Iter<'s, T> {
    pub(crate) fn new(iter: std::slice::Iter<'s, T>, shape: &'s Shape) -> Self {
        Self {
            iter,
            position: 0,
            shape,
            _marker: Default::default(),
        }
    }

    pub fn indexed(self) -> Indexed<'s, Self> {
        Indexed::new(self)
    }
}

impl<'s, T> Iterator for Iter<'s, T> {
    type Item = &'s T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize where Self: Sized {
        self.iter.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> where Self: Sized {
        self.iter.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n)
    }

    #[inline]
    fn for_each<F>(self, f: F) where Self: Sized, F: FnMut(Self::Item) {
        self.iter.for_each(f)
    }

    #[inline]
    fn collect<B: FromIterator<Self::Item>>(self) -> B where Self: Sized {
        self.iter.collect()
    }

    #[inline]
    fn partition<B, F>(self, f: F) -> (B, B) where Self: Sized, B: Default + Extend<Self::Item>, F: FnMut(&Self::Item) -> bool {
        self.iter.partition(f)
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B where Self: Sized, F: FnMut(B, Self::Item) -> B {
        self.iter.fold(init, f)
    }

    #[inline]
    fn reduce<F>(self, f: F) -> Option<Self::Item> where Self: Sized, F: FnMut(Self::Item, Self::Item) -> Self::Item {
        self.iter.reduce(f)
    }

    #[inline]
    fn all<F>(&mut self, f: F) -> bool where Self: Sized, F: FnMut(Self::Item) -> bool {
        self.iter.all(f)
    }

    #[inline]
    fn any<F>(&mut self, f: F) -> bool where Self: Sized, F: FnMut(Self::Item) -> bool {
        self.iter.any(f)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item> where Self: Sized, P: FnMut(&Self::Item) -> bool {
        self.iter.find(predicate)
    }

    #[inline]
    fn find_map<B, F>(&mut self, f: F) -> Option<B> where Self: Sized, F: FnMut(Self::Item) -> Option<B> {
        self.iter.find_map(f)
    }

    #[inline]
    fn position<P>(&mut self, predicate: P) -> Option<usize> where Self: Sized, P: FnMut(Self::Item) -> bool {
        self.iter.position(predicate)
    }

    #[inline]
    fn max(self) -> Option<Self::Item> where Self: Sized, Self::Item: Ord {
        self.iter.max()
    }

    #[inline]
    fn min(self) -> Option<Self::Item> where Self: Sized, Self::Item: Ord {
        self.iter.min()
    }

    #[inline]
    fn max_by_key<B: Ord, F>(self, f: F) -> Option<Self::Item> where Self: Sized, F: FnMut(&Self::Item) -> B {
        self.iter.max_by_key(f)
    }

    #[inline]
    fn max_by<F>(self, compare: F) -> Option<Self::Item> where Self: Sized, F: FnMut(&Self::Item, &Self::Item) -> Ordering {
        self.iter.max_by(compare)
    }

    #[inline]
    fn min_by_key<B: Ord, F>(self, f: F) -> Option<Self::Item> where Self: Sized, F: FnMut(&Self::Item) -> B {
        self.iter.min_by_key(f)
    }

    #[inline]
    fn min_by<F>(self, compare: F) -> Option<Self::Item> where Self: Sized, F: FnMut(&Self::Item, &Self::Item) -> Ordering {
        self.iter.min_by(compare)
    }

    #[inline]
    fn sum<S>(self) -> S where Self: Sized, S: Sum<Self::Item> {
        self.iter.sum()
    }

    #[inline]
    fn product<P>(self) -> P where Self: Sized, P: Product<Self::Item> {
        self.iter.product()
    }

    #[inline]
    fn cmp<I>(self, other: I) -> Ordering where I: IntoIterator<Item=Self::Item>, Self::Item: Ord, Self: Sized {
        self.iter.cmp(other)
    }
}

impl<'s, T> TensorIterator<'s> for Iter<'s, T> {
    fn position(&self) -> usize {
        self.position
    }

    fn get_shape(&self) -> &Shape {
        &self.shape
    }
}