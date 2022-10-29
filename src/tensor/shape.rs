use std::ops::Index;

#[derive(Clone, Debug, Eq)]
pub struct Shape {
    dimensions: Vec<usize>,
    capacity: usize,
    strides: Vec<usize>,
}

impl Shape {
    pub const fn zero() -> Self {
        Self {
            dimensions: Vec::new(),
            capacity: 1,
            strides: Vec::new(),
        }
    }

    pub fn new<S: AsRef<[usize]>>(dimensions: S) -> Self {
        let dimensions = dimensions.as_ref();
        let mut strides = vec![1; dimensions.len()];
        for i in (0..dimensions.len() - 1).rev() {
            strides[i] = dimensions[i + 1] * strides[i + 1]
        }
        Self {
            dimensions: Vec::from(dimensions),
            capacity: dimensions.iter().product(),
            strides,
        }
    }

    pub(crate) fn index(&self, index: &[usize]) -> Option<usize> {
        if index.len() != self.dimensions.len() {
            return None;
        }
        // let mut p = 0;
        // let mut s = 1;
        // for i in (0..self.dimensions.len()).rev() {
        //     if index[i] >= self.dimensions[i] {
        //         return None;
        //     }
        //     p += index[i] * s;
        //     s *= self.dimensions[i];
        // }
        // Some(p)
        let mut p = 0;
        for i in 0..self.dimensions.len() {
            if index[i] >= self.dimensions[i] {
                return None;
            }
            p += index[i] * self.strides[i];
        }
        Some(p)
    }

    pub(crate) fn inverse_index(&self, mut index: usize) -> Vec<usize> {
        let mut p = vec![0; self.dimensions.len()];
        for i in (0..self.dimensions.len()).rev() {
            p[i] = index % self.dimensions[i];
            index /= self.dimensions[i];
        }
        p
    }

    pub(crate) fn increment_index(&self, index: &mut [usize], mut inc: usize) -> bool {
        for i in (0..self.dimensions.len()).rev() {
            let t = index[i] + inc;
            index[i] = t % self.dimensions[i];
            inc = t / self.dimensions[i];
            if inc == 0 {
                return true;
            }
        }
        false
    }

    #[allow(dead_code)]
    pub(crate) fn decrement_index(&self, index: &mut [usize], mut dec: usize) -> bool {
        for i in (0..self.dimensions.len()).rev() {
            let t = self.dimensions[i] + index[i] - dec % self.dimensions[i];
            index[i] = t % self.dimensions[i];
            dec = (dec + t / self.dimensions[i]) / self.dimensions[i];
            if dec == 0 {
                return true;
            }
        }
        false
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions.len()
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.dimensions == other.dimensions
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dimensions[index]
    }
}

impl<U: AsRef<[usize]>> From<U> for Shape {
    fn from(shape: U) -> Self {
        Shape::new(shape)
    }
}
