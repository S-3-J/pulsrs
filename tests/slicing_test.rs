use pulsars::prelude::*;

#[test]
fn test_slice_rank_mismatch_fewer_ranges() {
    let t = Tensor::<f32>::zeros(vec![2, 3, 4]).unwrap(); // 3D tensor
    // Providing only 2 ranges for 3D tensor should error
    let result = t.slice(vec![(0..1).iosr(), (0..2).iosr()]);
    assert!(result.is_err());
}

#[test]
fn test_slice_rank_mismatch_more_ranges() {
    let t = Tensor::<f32>::zeros(vec![2, 3, 4]).unwrap(); // 3D tensor
    // Providing 4 ranges for 3D tensor should error
    let result = t.slice(vec![
        (0..1).iosr(), 
        (0..2).iosr(), 
        (0..3).iosr(),
        (0..1).iosr() // Extra range
    ]);
    assert!(result.is_err());
}

#[test]
fn test_slice_basic() {
    let t = Tensor::<f32>::arange(1.0, 12.0, 1.0).reshape(vec![3, 4]).unwrap();
    // Slice first row, first two columns
    let s = t.slice(vec![(0..1).iosr(), (0..2).iosr()]).unwrap();
    assert_eq!(s.shape().dims(), &[1, 2]);
    assert_eq!(s.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(s.get(&[0, 1]).unwrap(), 2.0);
}

#[test]
fn test_slice_with_none() {
    let t = Tensor::<f32>::arange(1.0, 12.0, 1.0).reshape(vec![3, 4]).unwrap();
    // Slice first row (index 0), first two columns take element at index 0
    let s = t.slice(vec![0.iosr(), (0..2).iosr()]).unwrap();
    assert_eq!(s.shape().dims(), &[1, 2]);
    assert_eq!(s.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(s.get(&[0, 1]).unwrap(), 2.0);
}

#[test]
fn test_slice_step() {
    let t = Tensor::<f32>::arange(1.0, 10.0, 1.0).reshape(vec![2, 5]).unwrap();
    // Slice first row (index 0 ), with step 2: take every other element
    let s = t.slice(vec![0.iosr(), ((0..5), 2).iosr()]).unwrap();
    assert_eq!(s.shape().dims(), &[1, 3]); // None takes index 0 from dim0 (size=1), step 2 from dim1 (0,2,4 -> 3 elements)
    assert_eq!(s.get(&[0, 0]).unwrap(), 1.0); // original[0,0] = 0
    assert_eq!(s.get(&[0, 1]).unwrap(), 3.0); // original[0,2] = 2
    assert_eq!(s.get(&[0, 2]).unwrap(), 5.0); // original[0,4] = 4
}

#[test]
fn test_slice_out_of_bounds() {
    let t = Tensor::<f32>::arange(1.0, 9.0, 1.0).reshape(vec![3, 3]).unwrap();
    // Start beyond dimension size
    let result = t.slice(vec![(5..10).iosr(), None]);
    assert!(result.is_err());
    // End beyond dimension size should be clamped
    let s = t.slice(vec![(0..5).iosr(), 0.iosr()]).unwrap(); // Should be clamped to (0..3) for first dim, 0 from second dim
    assert_eq!(s.shape().dims(), &[3, 1]);
    // Check some values: first dim all rows (0,1,2), second dim only index 0
    assert_eq!(s.get(&[0, 0]).unwrap(), 1.0); // original[0,0] = 0
    assert_eq!(s.get(&[1, 0]).unwrap(), 4.0); // original[1,0] = 3
    assert_eq!(s.get(&[2, 0]).unwrap(), 7.0); // original[2,0] = 6
}