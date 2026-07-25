#[allow(unused_imports)]
use pulsars::{
    buffer::Buffer,
    shape::Shape,
    stride::Stride,
    tensor::Tensor,
    error::PulsrsError
};

// =============================================================================
// SCALAR TENSOR TESTS
// =============================================================================

#[test]
fn test_tensor_creation_for_scalar() {
    let x: Tensor<f32> = 3.0.into();

    assert!(x.is_scalar());
    assert_eq!(x.shape().dims(), &[]);
    assert_eq!(x.strides(), &[]);
    assert_eq!(x.get(&[]).unwrap(), 3.0);
}

#[test]
fn test_scalar_get_rejects_indices() {
    let x: Tensor<f32> = 3.0.into();
    let result = x.get(&[0]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::TensorIndexingError { shape, indices } => {
            assert!(shape.is_scalar());
            assert_eq!(indices, vec![0]);
        }
        _ => panic!("Expected TensorIndexingError"),
    }
}

#[test]
fn test_scalar_get_accepts_empty_indices() {
    let x: Tensor<f32> = 42.0.into();
    assert_eq!(x.get(&[]).unwrap(), 42.0);
}

#[test]
fn test_scalar_is_always_contiguous() {
    let x: Tensor<i32> = (-5).into();
    assert!(x.is_contiguous());
}

// =============================================================================
// VECTOR TENSOR TESTS
// =============================================================================

#[test]
fn test_tensor_creation_from_vector() {
    let x: Tensor<f32> = Tensor::from_vector(vec![2.6f32, 3.4, 5.6, 5.7, 4.6, 33.5], vec![6]).unwrap();

    assert!(!x.is_scalar());
    assert_eq!(x.shape().dims(), &[6]);
    assert_eq!(x.strides(), &[1]);
    assert_eq!(x.get(&[2]).unwrap(), 5.6);
}

#[test]
fn test_vector_get_at_boundaries() {
    let x: Tensor<i32> = Tensor::from_vector(vec![1, 2, 3, 4, 5], vec![5]).unwrap();

    assert_eq!(x.get(&[0]).unwrap(), 1);
    assert_eq!(x.get(&[4]).unwrap(), 5);
}

#[test]
fn test_vector_get_out_of_bounds() {
    let x: Tensor<i32> = Tensor::from_vector(vec![1, 2, 3], vec![3]).unwrap();

    let result = x.get(&[3]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::TensorIndexingError { indices, .. } => {
            assert_eq!(indices, vec![3]);
        }
        _ => panic!("Expected TensorIndexingError"),
    }
}

#[test]
fn test_vector_get_wrong_rank() {
    let x: Tensor<i32> = Tensor::from_vector(vec![1, 2, 3], vec![3]).unwrap();

    let result = x.get(&[1, 0]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::RankMismatch { expected, found } => {
            assert_eq!(expected, 1);
            assert_eq!(found, 2);
        }
        _ => panic!("Expected RankMismatch"),
    }
}

#[test]
fn test_vector_from_wrong_element_count() {
    let result = Tensor::<i32>::from_vector(vec![1, 2, 3], vec![5]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::ShapeIncompatiblewithElements { expected, permitted } => {
            assert_eq!(expected, 3);
            assert_eq!(permitted, 5);
        }
        _ => panic!("Expected ShapeIncompatiblewithElements"),
    }
}

// =============================================================================
// 2D TENSOR TESTS
// =============================================================================

#[test]
fn test_2d_tensor_get() {
    let x: Tensor<f32> = Tensor::from_vector(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3]
    ).unwrap();

    assert_eq!(x.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(x.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(x.get(&[0, 2]).unwrap(), 3.0);
    assert_eq!(x.get(&[1, 0]).unwrap(), 4.0);
    assert_eq!(x.get(&[1, 1]).unwrap(), 5.0);
    assert_eq!(x.get(&[1, 2]).unwrap(), 6.0);
}

#[test]
fn test_2d_tensor_get_out_of_bounds() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    assert!(x.get(&[2, 0]).is_err());
    assert!(x.get(&[0, 2]).is_err());
    assert!(x.get(&[2, 2]).is_err());
}

#[test]
fn test_2d_tensor_strides() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    assert_eq!(x.strides(), &[2, 1]);
}

#[test]
fn test_2d_tensor_numel() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    assert_eq!(x.numel(), 6);
}

// =============================================================================
// 3D+ TENSOR TESTS
// =============================================================================

#[test]
fn test_3d_tensor_get() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![2, 2, 2]
    ).unwrap();

    assert_eq!(x.get(&[0, 0, 0]).unwrap(), 1);
    assert_eq!(x.get(&[0, 0, 1]).unwrap(), 2);
    assert_eq!(x.get(&[0, 1, 0]).unwrap(), 3);
    assert_eq!(x.get(&[0, 1, 1]).unwrap(), 4);
    assert_eq!(x.get(&[1, 0, 0]).unwrap(), 5);
    assert_eq!(x.get(&[1, 0, 1]).unwrap(), 6);
    assert_eq!(x.get(&[1, 1, 0]).unwrap(), 7);
    assert_eq!(x.get(&[1, 1, 1]).unwrap(), 8);
}

#[test]
fn test_3d_tensor_strides() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 1, 3]
    ).unwrap();

    assert_eq!(x.strides(), &[3, 3, 1]);
}

#[test]
fn test_4d_tensor_basic() {
    let x: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 2, 2, 3]
    ).unwrap();

    assert_eq!(x.rank(), 4);
    assert_eq!(x.numel(), 24);
    assert_eq!(x.get(&[0, 0, 0, 0]).unwrap(), 0);
    assert_eq!(x.get(&[1, 1, 1, 2]).unwrap(), 23);
}

// =============================================================================
// SINGLETON DIMENSION TESTS
// =============================================================================

#[test]
fn test_singleton_dimension_1d() {
    let x: Tensor<i32> = Tensor::from_vector(vec![42], vec![1]).unwrap();
    assert_eq!(x.numel(), 1);
    assert_eq!(x.get(&[0]).unwrap(), 42);
}

#[test]
fn test_singleton_dimension_2d() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3],
        vec![1, 3]
    ).unwrap();

    assert_eq!(x.numel(), 3);
    assert_eq!(x.get(&[0, 0]).unwrap(), 1);
    assert_eq!(x.get(&[0, 2]).unwrap(), 3);
}

#[test]
fn test_trailing_singleton_dimension() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2],
        vec![2, 1]
    ).unwrap();

    assert_eq!(x.numel(), 2);
    assert_eq!(x.get(&[0, 0]).unwrap(), 1);
    assert_eq!(x.get(&[1, 0]).unwrap(), 2);
}

// =============================================================================
// ZERO DIMENSION TESTS
// =============================================================================

#[test]
fn test_zero_element_tensor() {
    let result = Tensor::<i32>::from_vector(vec![], vec![0]);
    assert!(result.is_ok());
    let x = result.unwrap();
    assert_eq!(x.numel(), 0);
}

#[test]
fn test_zero_element_tensor_get() {
    let x: Tensor<i32> = Tensor::from_vector(vec![], vec![0]).unwrap();
    assert!(x.get(&[0]).is_err());
}

#[test]
fn test_zero_in_shape() {
    let result = Tensor::<i32>::from_vector(vec![], vec![3, 0, 4]);
    assert!(result.is_ok());
    let x = result.unwrap();
    assert_eq!(x.numel(), 0);
}

// =============================================================================
// CONTIGUOUS FLAG TESTS
// =============================================================================

#[test]
fn test_tensor_contiguous() {
    let x: Tensor<f32> = Tensor::from_vector(vec![2.6f32, 3.4, 5.6, 5.7, 4.6, 33.5], vec![6]).unwrap();
    let r_x = x.reshape(vec![2,3]).unwrap();
    let r_x_transpose = r_x.permute(vec![1, 0]).unwrap();

    assert!(x.is_contiguous());
    assert!(r_x.is_contiguous());
    assert!(!r_x_transpose.is_contiguous());
}

#[test]
fn test_identity_permutation_is_contiguous() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let identity = x.permute(vec![0, 1]).unwrap();
    assert!(identity.is_contiguous());
}

#[test]
fn test_transpose_2d_is_not_contiguous() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let transposed = x.permute(vec![1, 0]).unwrap();
    assert!(!transposed.is_contiguous());
}

#[test]
fn test_scalar_always_contiguous() {
    let x: Tensor<i32> = 42.into();
    assert!(x.is_contiguous());
}

// =============================================================================
// RESHAPE TESTS
// =============================================================================

#[test]
fn test_reshape_basic() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let reshaped = x.reshape(vec![3, 2]).unwrap();
    assert_eq!(reshaped.shape().dims(), &[3, 2]);
    assert_eq!(reshaped.get(&[0, 0]).unwrap(), 1);
    assert_eq!(reshaped.get(&[2, 1]).unwrap(), 6);
}

#[test]
fn test_reshape_to_1d() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let flat = x.reshape(vec![4]).unwrap();
    assert_eq!(flat.shape().dims(), &[4]);
    assert_eq!(flat.get(&[0]).unwrap(), 1);
    assert_eq!(flat.get(&[3]).unwrap(), 4);
}

#[test]
fn test_reshape_with_singleton() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let reshaped = x.reshape(vec![1, 4]).unwrap();
    assert_eq!(reshaped.shape().dims(), &[1, 4]);
    assert_eq!(reshaped.get(&[0, 3]).unwrap(), 4);
}

#[test]
fn test_reshape_preserves_numel() {
    let x: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 3, 4]
    ).unwrap();

    let reshaped = x.reshape(vec![4, 6]).unwrap();
    assert_eq!(reshaped.numel(), 24);
}

#[test]
fn test_reshape_wrong_element_count() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let result = x.reshape(vec![3, 3]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::ReshapedElementMismatch { expected, found } => {
            assert_eq!(expected, 4);
            assert_eq!(found, 9);
        }
        _ => panic!("Expected ReshapedElementMismatch"),
    }
}

#[test]
fn test_reshape_to_scalar() {
    let x: Tensor<i32> = Tensor::from_vector(vec![42], vec![1]).unwrap();
    let scalar = x.reshape(vec![]).unwrap();
    assert!(scalar.is_scalar());
    assert_eq!(scalar.get(&[]).unwrap(), 42);
}

// =============================================================================
// PERMUTE TESTS
// =============================================================================

#[test]
fn test_permute_2d_transpose() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let transposed = x.permute(vec![1, 0]).unwrap();
    assert_eq!(transposed.get(&[0, 0]).unwrap(), 1);
    assert_eq!(transposed.get(&[0, 1]).unwrap(), 3);
    assert_eq!(transposed.get(&[1, 0]).unwrap(), 2);
    assert_eq!(transposed.get(&[1, 1]).unwrap(), 4);
}

#[test]
fn test_permute_3d() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![1, 2, 3]
    ).unwrap();

    let perm = x.permute(vec![2, 0, 1]).unwrap();
    assert_eq!(perm.shape().dims(), &[3, 1, 2]);
    assert_eq!(perm.strides()[0], 1);
}

#[test]
fn test_permute_axis_out_of_bounds() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let result = x.permute(vec![0, 2]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::AxisOutofBounds { max, min: _, found } => {
            assert_eq!(max, 2);
            assert_eq!(found, 2);
        }
        _ => panic!("Expected AxisOutofBounds"),
    }
}

#[test]
fn test_permute_duplicate_axis() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let result = x.permute(vec![0, 0]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::UniqueDataConstraintError { given } => {
            assert_eq!(given, vec![0, 0]);
        }
        _ => panic!("Expected UniqueDataConstraintError"),
    }
}

#[test]
fn test_permute_wrong_length() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let result = x.permute(vec![0]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::RankMismatch { expected, found } => {
            assert_eq!(expected, 2);
            assert_eq!(found, 1);
        }
        _ => panic!("Expected RankMismatch"),
    }
}

#[test]
fn test_permute_scalar_rejects_non_empty() {
    let x: Tensor<i32> = 42.into();
    let result = x.permute(vec![0]);
    assert!(result.is_err());
    match result.unwrap_err() {
        PulsrsError::RankMismatch { expected, found } => {
            assert_eq!(expected, 0);
            assert_eq!(found, 1);
        }
        _ => panic!("Expected RankMismatch"),
    }
}

#[test]
fn test_permute_scalar_accepts_empty() {
    let x: Tensor<i32> = 42.into();
    let result = x.permute(vec![]);
    assert!(result.is_ok());
    assert!(result.unwrap().is_scalar());
}

// =============================================================================
// FLATTEN TESTS
// =============================================================================

#[test]
fn test_flatten_2d() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let flat = x.flatten().unwrap();
    assert_eq!(flat.shape().dims(), &[6]);
    assert_eq!(flat.get(&[0]).unwrap(), 1);
    assert_eq!(flat.get(&[5]).unwrap(), 6);
}

#[test]
fn test_flatten_3d() {
    let x: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 3, 4]
    ).unwrap();

    let flat = x.flatten().unwrap();
    assert_eq!(flat.shape().dims(), &[24]);
    assert_eq!(flat.get(&[0]).unwrap(), 0);
    assert_eq!(flat.get(&[23]).unwrap(), 23);
}

#[test]
fn test_flatten_already_1d() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3],
        vec![3]
    ).unwrap();

    let flat = x.flatten().unwrap();
    assert_eq!(flat.shape().dims(), &[3]);
}

#[test]
fn test_flatten_scalar() {
    let x: Tensor<i32> = 42.into();
    let flat = x.flatten().unwrap();
    assert_eq!(flat.shape().dims(), &[1]);
    assert_eq!(flat.get(&[0]).unwrap(), 42);
}

// =============================================================================
// CONTIGUOUS OPERATION TESTS
// =============================================================================

#[test]
fn test_contiguous_preserves_data_2d() {
    let original: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let transposed = original.permute(vec![1, 0]).unwrap();
    let made_contiguous = transposed.contiguous();

    assert!(made_contiguous.is_contiguous());
    assert_eq!(made_contiguous.get(&[0, 0]).unwrap(), 1);
    assert_eq!(made_contiguous.get(&[0, 1]).unwrap(), 4);
    assert_eq!(made_contiguous.get(&[1, 0]).unwrap(), 2);
    assert_eq!(made_contiguous.get(&[1, 1]).unwrap(), 5);
    assert_eq!(made_contiguous.get(&[2, 0]).unwrap(), 3);
    assert_eq!(made_contiguous.get(&[2, 1]).unwrap(), 6);
}

#[test]
fn test_contiguous_does_not_modify_original() {
    let original: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let transposed = original.permute(vec![1, 0]).unwrap();
    let _contiguous = transposed.contiguous();

    assert!(!transposed.is_contiguous());
}

#[test]
fn test_contiguous_already_contiguous_returns_clone() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let result = x.contiguous();
    assert!(result.is_contiguous());
}

#[test]
fn test_contiguous_preserves_numel() {
    let x: Tensor<i32> = Tensor::from_vector(
        (0..120).collect(),
        vec![2, 3, 4, 5]
    ).unwrap();

    let transposed = x.permute(vec![3, 2, 1, 0]).unwrap();
    let made_contiguous = transposed.contiguous();

    assert_eq!(made_contiguous.numel(), 120);
}

#[test]
fn test_contiguous_scalar() {
    let x: Tensor<i32> = 42.into();
    let result = x.contiguous();
    assert!(result.is_contiguous());
    assert_eq!(result.get(&[]).unwrap(), 42);
}

// =============================================================================
// CHAINED OPERATION TESTS
// =============================================================================

#[test]
fn test_reshape_permute_get_chain() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![2, 2, 2]
    ).unwrap();

    let reshaped = x.reshape(vec![4, 2]).unwrap();
    let permuted = reshaped.permute(vec![1, 0]).unwrap();
    let contiguous = permuted.contiguous();

    assert_eq!(contiguous.get(&[0, 0]).unwrap(), 1);
    assert_eq!(contiguous.get(&[0, 1]).unwrap(), 3);
    assert_eq!(contiguous.get(&[1, 0]).unwrap(), 2);
    assert_eq!(contiguous.get(&[1, 1]).unwrap(), 4);
}

#[test]
fn test_flatten_reshape_get_chain() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let flat = x.flatten().unwrap();
    let reshaped = flat.reshape(vec![3, 2]).unwrap();

    assert_eq!(reshaped.get(&[0, 0]).unwrap(), 1);
    assert_eq!(reshaped.get(&[2, 1]).unwrap(), 6);
}

#[test]
fn test_multiple_permutes() {
    let x: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 3, 4]
    ).unwrap();

    let p1 = x.permute(vec![2, 1, 0]).unwrap();
    let p2 = p1.permute(vec![2, 1, 0]).unwrap();

    assert_eq!(p2.get(&[0, 0, 0]).unwrap(), 0);
    assert_eq!(p1.get(&[3, 2, 1]).unwrap(), 23);
}

#[test]
fn test_permute_contiguous_reshape() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let transposed = x.permute(vec![1, 0]).unwrap();
    let c = transposed.contiguous();
    let r = c.reshape(vec![3, 2]).unwrap();

    assert_eq!(r.get(&[0, 0]).unwrap(), 1);
    assert_eq!(r.get(&[2, 1]).unwrap(), 6);
}

// =============================================================================
// METADATA INVARIANT TESTS
// =============================================================================

#[test]
fn test_shape_stride_rank_consistency_after_creation() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    assert_eq!(x.rank(), x.shape().rank());
    assert_eq!(x.rank(), x.strides().len());
    assert_eq!(x.ndim(), x.rank());
}

#[test]
fn test_shape_stride_rank_consistency_after_reshape() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let r = x.reshape(vec![3, 2]).unwrap();
    assert_eq!(r.rank(), r.shape().rank());
    assert_eq!(r.rank(), r.strides().len());
}

#[test]
fn test_shape_stride_rank_consistency_after_permute() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let p = x.permute(vec![1, 0]).unwrap();
    assert_eq!(p.rank(), p.shape().rank());
    assert_eq!(p.rank(), p.strides().len());
}

#[test]
fn test_contiguous_flag_matches_actual_layout() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    assert!(x.is_contiguous());

    let t = x.permute(vec![1, 0]).unwrap();
    assert!(!t.is_contiguous());

    let c = t.contiguous();
    assert!(c.is_contiguous());
}

#[test]
fn test_reshaped_tensor_preserves_numel() {
    let original: Tensor<i32> = Tensor::from_vector(
        (0..60).collect(),
        vec![3, 4, 5]
    ).unwrap();

    let reshaped = original.reshape(vec![4, 3, 5]).unwrap();
    assert_eq!(reshaped.numel(), original.numel());
    assert_eq!(reshaped.numel(), 60);
}

#[test]
fn test_permuted_tensor_preserves_numel() {
    let original: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 3, 4]
    ).unwrap();

    let permuted = original.permute(vec![2, 0, 1]).unwrap();
    assert_eq!(permuted.numel(), original.numel());
}

#[test]
fn test_strides_consistent_with_shape() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let shape = x.shape().dims();
    let strides = x.strides();

    for i in 0..strides.len() {
        let expected_stride: usize = shape[i+1..].iter().product();
        assert_eq!(strides[i], expected_stride);
    }
}

// =============================================================================
// LARGE TENSOR TESTS
// =============================================================================

#[test]
fn test_large_1d_tensor() {
    let data: Vec<i32> = (0..10000).collect();
    let x: Tensor<i32> = Tensor::from_vector(data.clone(), vec![10000]).unwrap();

    assert_eq!(x.numel(), 10000);
    assert_eq!(x.get(&[0]).unwrap(), 0);
    assert_eq!(x.get(&[9999]).unwrap(), 9999);
}

#[test]
fn test_large_2d_tensor() {
    let data: Vec<i32> = (0..1000).collect();
    let x: Tensor<i32> = Tensor::from_vector(data, vec![10, 100]).unwrap();

    assert_eq!(x.numel(), 1000);
    assert_eq!(x.get(&[0, 0]).unwrap(), 0);
    assert_eq!(x.get(&[9, 99]).unwrap(), 999);
}

#[test]
fn test_large_tensor_reshape() {
    let data: Vec<i32> = (0..1000).collect();
    let x: Tensor<i32> = Tensor::from_vector(data, vec![10, 100]).unwrap();

    let reshaped = x.reshape(vec![20, 50]).unwrap();
    assert_eq!(reshaped.numel(), 1000);
    assert_eq!(reshaped.get(&[19, 49]).unwrap(), 999);
}

#[test]
fn test_large_tensor_contiguous() {
    let data: Vec<i32> = (0..1000).collect();
    let x: Tensor<i32> = Tensor::from_vector(data, vec![10, 100]).unwrap();

    let t = x.permute(vec![1, 0]).unwrap();
    let c = t.contiguous();

    assert!(c.is_contiguous());
    assert_eq!(c.get(&[0, 0]).unwrap(), 0);
    assert_eq!(c.get(&[99, 9]).unwrap(), 999);
}

// =============================================================================
// VIEW SEMANTICS TESTS
// =============================================================================

#[test]
fn test_permute_shares_buffer() {
    let original: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let permuted = original.permute(vec![1, 0]).unwrap();

    assert_eq!(original.get(&[0, 0]).unwrap(), 1);
    assert_eq!(permuted.get(&[0, 0]).unwrap(), 1);
}

// =============================================================================
// EDGE CASE TESTS
// =============================================================================

#[test]
fn test_high_rank_tensor_7d() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2],
        vec![1, 1, 1, 1, 1, 1, 2]
    ).unwrap();

    assert_eq!(x.rank(), 7);
    assert_eq!(x.numel(), 2);
}

#[test]
fn test_vector_stride_calculation() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![6]
    ).unwrap();

    assert_eq!(x.strides(), &[1]);
}

#[test]
fn test_strides_with_zeros_in_shape() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![],
        vec![0, 5]
    ).unwrap();

    assert_eq!(x.numel(), 0);
    assert_eq!(x.strides(), &[5, 1]);
}

#[test]
fn test_all_ones_shape() {
    let x: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![1, 1, 1, 1, 1, 1, 1, 8]
    ).unwrap();

    assert_eq!(x.numel(), 8);
    assert_eq!(x.get(&[0, 0, 0, 0, 0, 0, 0, 7]).unwrap(), 8);
}