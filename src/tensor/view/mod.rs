use crate::tensor::SliceIndex;

pub mod view_mut;
pub mod view;
// todo: make views better

pub fn get_index<I: AsRef<[usize]>>(slices: &[SliceIndex], index: I) -> Vec<usize> {
    let index = index.as_ref();
    let mut k = 0;
    slices.iter().map(|&i| match i {
        SliceIndex::Single(t) => t,
        SliceIndex::Range(a, b, c) => {
            let a = a.unwrap_or_default();
            let c = c.unwrap_or(1);
            let t = index[k] * c + a;
            if let Some(b) = b {
                assert!(t < b)
            }
            k += 1;
            t
        }
    }).collect()
}
