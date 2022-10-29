use cognitio::tensor::{SliceIndex, Tensor};

fn main() {
    let t = Tensor::from([
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ],
        [
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
        ],
    ]);
    // for i in 0..3 {
    //     println!("{}", t[[i, 0, 0]]);
    // }
    let s = t.slice(vec![
        SliceIndex::Range(None, Some(2), None),
        SliceIndex::Single(1),
        SliceIndex::Range(None, None, Some(2)),
    ]);
    for i in 0..2 {
        for j in 0..2 {
            print!("{} ", s[[i, j]]);
        }
        println!()
    }
    println!("{:?}", Tensor::from([[[[[[()]]]]]])[[0, 0, 0, 0, 0, 0]]);
}
