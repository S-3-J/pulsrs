
use pulsars::shape::Shape;
use pulsars::stride::Stride;
use pulsars::error::PulsrsError;

#[test]
fn test_stride_construction_from_shape() {
    let shape : Shape = vec![3,4,5].into();

    let stride = Stride::from_shape(&shape);

    assert_eq!(shape.rank(), stride.rank());
}

#[test]
fn test_stride_strides() {
    let shape : Shape = vec![3,4,5].into();

    let stride = Stride::from_shape(&shape);

    assert_eq!(stride.strides(), &[20, 5, 1]);
}

#[test]
fn test_stride_get() {
    let shape : Shape = vec![3,4,5].into();

    let stride = Stride::from_shape(&shape);

    assert_eq!(stride.get(0), Some(20));
}

#[test]
fn test_stride_offset() {
    let shape : Shape = vec![3,4,5].into();

    let stride = Stride::from_shape(&shape);

    assert_eq!(stride.offset(&vec![1,2,3]).unwrap_or(0), 33)
}

#[test]
fn test_stride_for_scalar() {
    let shape : Shape = Shape::default();

    let stride = Stride::from_shape(&shape);

    assert_eq!(stride.rank(), 0)
}


#[test]
fn test_stride_for_offset_error() {
    let shape: Shape = vec![3,4,5].into();

    let stride = Stride::from_shape(&shape);
    
    assert_eq!(stride.offset(&[1,2,3,4]).expect_err("Something went wrong in errors"), PulsrsError::RankMismatch { expected: 3, found: 4 })
}