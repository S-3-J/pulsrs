
use pulsars::{
    buffer::buffer::Buffer,
    shape::shape::Shape,
    stride::stride::Stride,
    tensor::tensor::Tensor,
    error::error::PulsrsError
};

#[test]
fn test_tensor_creation_for_scalar() {
    let x: Tensor<f32> = 3.0.into();

    assert!(x.is_scalar());
    assert_eq!(x.shape().dims(), &[]);
    assert_eq!(x.strides(), &[]);
    assert_eq!(x.get(&[]).unwrap(), 3.0);

}

#[test]
fn test_tensor_creation_from_vector() {
    let x: Tensor<f32> = Tensor::from_vector(vec![2.6f32, 3.4, 5.6, 5.7, 4.6, 33.5], vec![6]).unwrap();

    assert!(!x.is_scalar());
    assert_eq!(x.shape().dims(), &[6]);
    assert_eq!(x.strides(), &[1]);
    assert_eq!(x.get(&[2]).unwrap(), 5.6);
}

#[test]
fn test_tensor_contiguous() {
    let x: Tensor<f32> = Tensor::from_vector(vec![2.6f32, 3.4, 5.6, 5.7, 4.6, 33.5], vec![6]).unwrap();
    let r_x = x.reshape(vec![2,3]).unwrap();
    let r_x_transpose = r_x.permute(vec![1, 0]).unwrap();

    assert!(x.is_contiguous());
    assert!(r_x.is_contiguous());
    assert!(!r_x_transpose.is_contiguous());
}
