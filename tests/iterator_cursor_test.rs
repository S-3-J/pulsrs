
#[allow(unused_imports)]
use pulsars::{
    cursor::IndexCursor,
    iterator::{
        iteration_style::IterStyle,
        IndexIterator,
        TensorIterator,
    },
    tensor::Tensor,
    error::PulsrsError
};

// =============================================================================
// INDEX CURSOR TESTS
// =============================================================================

#[test]
fn test_index_cursor_scalar_dims() {
    let mut cursor = IndexCursor::new(vec![], vec![]);
    assert!(!cursor.finished());
    cursor.advance();
    assert!(cursor.finished());
}

#[test]
fn test_index_cursor_basic_advance_c_style() {
    let mut cursor = IndexCursor::new(vec![2, 3], vec![1, 0]);

    assert_eq!(cursor.index(), &[0, 0]);
    cursor.advance();
    assert_eq!(cursor.index(), &[0, 1]);
    cursor.advance();
    assert_eq!(cursor.index(), &[0, 2]);
    cursor.advance();
    assert_eq!(cursor.index(), &[1, 0]);
    cursor.advance();
    assert_eq!(cursor.index(), &[1, 1]);
    cursor.advance();
    assert_eq!(cursor.index(), &[1, 2]);
    cursor.advance();
    assert!(cursor.finished());
}

#[test]
fn test_index_cursor_advance_f_style() {
    let mut cursor = IndexCursor::new(vec![2, 3], vec![0, 1]);

    assert_eq!(cursor.index(), &[0, 0]);
    cursor.advance();
    assert_eq!(cursor.index(), &[1, 0]);
    cursor.advance();
    assert_eq!(cursor.index(), &[0, 1]);
    assert!(!cursor.finished());
}

#[test]
fn test_index_cursor_reset() {
    let mut cursor = IndexCursor::new(vec![2, 3], vec![1, 0]);

    cursor.advance();
    cursor.advance();
    assert!(!cursor.finished());

    cursor.reset();
    assert!(!cursor.finished());
    assert_eq!(cursor.index(), &[0, 0]);
}

#[test]
fn test_index_cursor_single_dimension() {
    let mut cursor = IndexCursor::new(vec![5], vec![0]);

    for i in 0..5 {
        assert_eq!(cursor.index(), &[i]);
        cursor.advance();
    }
    assert!(cursor.finished());
}

#[test]
fn test_index_cursor_three_dimensions() {
    let mut cursor = IndexCursor::new(vec![2, 3, 4], vec![2, 1, 0]);

    assert_eq!(cursor.index(), &[0, 0, 0]);
    cursor.advance();
    assert_eq!(cursor.index(), &[0, 0, 1]);
    cursor.advance();
    assert_eq!(cursor.index(), &[0, 0, 2]);
    cursor.advance();
    assert_eq!(cursor.index(), &[0, 0, 3]);
    cursor.advance();
    assert_eq!(cursor.index(), &[0, 1, 0]);
}

#[test]
fn test_index_cursor_total_iterations() {
    let dims = vec![2, 3, 4];
    let mut cursor = IndexCursor::new(dims.clone(), vec![2, 1, 0]);
    let mut count = 0;

    while !cursor.finished() {
        count += 1;
        cursor.advance();
    }

    let expected: usize = dims.iter().product();
    assert_eq!(count, expected);
}

// =============================================================================
// ITERSTYLE TESTS
// =============================================================================

#[test]
fn test_iter_style_c_style() {
    let style = IterStyle::Cstyle;
    let result = style.process(3);
    assert_eq!(result, vec![2, 1, 0]);
}

#[test]
fn test_iter_style_f_style() {
    let style = IterStyle::Fstyle;
    let result = style.process(3);
    assert_eq!(result, vec![0, 1, 2]);
}

#[test]
fn test_iter_style_custom() {
    let style = IterStyle::Custom(vec![2, 0, 1]);
    let result = style.process(3);
    assert_eq!(result, vec![2, 0, 1]);
}

#[test]
fn test_iter_style_c_style_single_dim() {
    let style = IterStyle::Cstyle;
    let result = style.process(1);
    assert_eq!(result, vec![0]);
}

#[test]
fn test_iter_style_f_style_single_dim() {
    let style = IterStyle::Fstyle;
    let result = style.process(1);
    assert_eq!(result, vec![0]);
}

// =============================================================================
// INDEX ITERATOR TESTS
// =============================================================================

#[test]
fn test_index_iterator_2d_c_style() {
    let cursor = IndexCursor::new(vec![2, 3], vec![1, 0]);
    let mut iter = IndexIterator::new(cursor);

    let expected = vec![
        vec![0, 0], vec![0, 1], vec![0, 2],
        vec![1, 0], vec![1, 1], vec![1, 2],
    ];

    for exp in expected {
        let idx = iter.next().unwrap();
        assert_eq!(idx, exp);
    }
    assert!(iter.next().is_none());
}

#[test]
fn test_index_iterator_2d_f_style() {
    let cursor = IndexCursor::new(vec![2, 3], vec![0, 1]);
    let mut iter = IndexIterator::new(cursor);

    let expected = vec![
        vec![0, 0], vec![1, 0],
        vec![0, 1], vec![1, 1],
        vec![0, 2], vec![1, 2],
    ];

    for exp in expected {
        let idx = iter.next().unwrap();
        assert_eq!(idx, exp);
    }
    assert!(iter.next().is_none());
}

#[test]
fn test_index_iterator_1d() {
    let cursor = IndexCursor::new(vec![4], vec![0]);
    let mut iter = IndexIterator::new(cursor);

    let expected = vec![vec![0], vec![1], vec![2], vec![3]];

    for exp in expected {
        let idx = iter.next().unwrap();
        assert_eq!(idx, exp);
    }
    assert!(iter.next().is_none());
}

#[test]
fn test_index_iterator_produces_correct_count() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 3, 4]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.index_iterator(style);
    let count = iter.count();

    assert_eq!(count, 24);
}

#[test]
fn test_index_iterator_with_custom_priority() {
    let style = IterStyle::Custom(vec![2, 0, 1]);
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..6).collect(),
        vec![1, 2, 3]
    ).unwrap();

    let mut iter = tensor.index_iterator(style);
    let first = iter.next().unwrap();
    assert_eq!(first, vec![0, 0, 0]);
}

// =============================================================================
// TENSOR ITERATOR TESTS
// =============================================================================

#[test]
fn test_tensor_iterator_1d() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![10, 20, 30, 40],
        vec![4]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);

    assert_eq!(iter.next(), Some(10));
    assert_eq!(iter.next(), Some(20));
    assert_eq!(iter.next(), Some(30));
    assert_eq!(iter.next(), Some(40));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_tensor_iterator_2d_c_style() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);

    let expected = vec![1, 2, 3, 4, 5, 6];
    for exp in expected {
        assert_eq!(iter.next(), Some(exp));
    }
    assert_eq!(iter.next(), None);
}

#[test]
fn test_tensor_iterator_2d_f_style() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let style = IterStyle::Fstyle;
    let mut iter = tensor.tensor_iterator(style);

    let expected = vec![1, 4, 2, 5, 3, 6];
    for exp in expected {
        assert_eq!(iter.next(), Some(exp));
    }
    assert_eq!(iter.next(), None);
}

#[test]
fn test_tensor_iterator_collect() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    assert_eq!(collected, vec![1, 2, 3, 4]);
}

#[test]
fn test_tensor_iterator_count() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..120).collect(),
        vec![2, 3, 4, 5]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    assert_eq!(iter.count(), 120);
}

#[test]
fn test_tensor_iterator_3d() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![2, 2, 2]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);

    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.next(), Some(2));
    assert_eq!(iter.next(), Some(3));
    assert_eq!(iter.next(), Some(4));
    assert_eq!(iter.next(), Some(5));
    assert_eq!(iter.next(), Some(6));
    assert_eq!(iter.next(), Some(7));
    assert_eq!(iter.next(), Some(8));
    assert_eq!(iter.next(), None);
}

// =============================================================================
// EDGE CASES - ITERATION
// =============================================================================

#[test]
fn test_iterator_scalar_tensor() {
    let tensor: Tensor<i32> = 42.into();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);

    assert_eq!(iter.next(), Some(42));
    assert!(iter.next().is_none());
}

#[test]
fn test_iterator_scalar_tensor_f_style() {
    let tensor: Tensor<i32> = 42.into();

    let style = IterStyle::Fstyle;
    let mut iter = tensor.tensor_iterator(style);

    assert_eq!(iter.next(), Some(42));
    assert!(iter.next().is_none());
}

#[test]
#[allow(unused_mut)]
fn test_iterator_zero_size_dimension() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![],
        vec![0, 3]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);
    assert_eq!(iter.count(), 0);
}

#[test]
fn test_iterator_single_element() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![99],
        vec![1]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);

    assert_eq!(iter.next(), Some(99));
    assert_eq!(iter.next(), None);
}

#[test]
fn test_iterator_singleton_dimensions() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3],
        vec![1, 1, 3]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    assert_eq!(collected, vec![1, 2, 3]);
}

// =============================================================================
// INDEX TRAIT TESTS
// =============================================================================

#[test]
fn test_tensor_index_trait_1d() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![10, 20, 30],
        vec![3]
    ).unwrap();

    assert_eq!(tensor[&[0]], 10);
    assert_eq!(tensor[&[1]], 20);
    assert_eq!(tensor[&[2]], 30);
}

#[test]
fn test_tensor_index_trait_2d() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    assert_eq!(tensor[&[0, 0]], 1);
    assert_eq!(tensor[&[0, 1]], 2);
    assert_eq!(tensor[&[1, 0]], 3);
    assert_eq!(tensor[&[1, 1]], 4);
}

#[test]
fn test_tensor_index_trait_scalar() {
    let tensor: Tensor<i32> = 42.into();
    assert_eq!(tensor[&[]], 42);
}

#[test]
#[should_panic]
fn test_tensor_index_trait_panics_on_oob() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3],
        vec![3]
    ).unwrap();

    let _ = tensor[&[10]];
}

#[test]
#[should_panic]
fn test_tensor_index_trait_panics_on_wrong_rank() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3],
        vec![3]
    ).unwrap();

    let _ = tensor[&[0, 0]];
}

// =============================================================================
// CHAINED OPERATIONS WITH ITERATORS
// =============================================================================

#[test]
fn test_iterate_then_verify_with_get() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![6]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    for (i, &val) in collected.iter().enumerate() {
        let idx = vec![i];
        assert_eq!(tensor.get(&idx).unwrap(), val);
    }
}

#[test]
fn test_iterate_after_reshape() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..12).collect(),
        vec![3, 4]
    ).unwrap();

    let reshaped = tensor.reshape(vec![4, 3]).unwrap();
    let style = IterStyle::Cstyle;
    let iter = reshaped.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    assert_eq!(collected.len(), 12);
    assert_eq!(collected, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
}

#[test]
fn test_iterate_after_permute() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let transposed = tensor.permute(vec![1, 0]).unwrap();
    let style = IterStyle::Cstyle;
    let iter = transposed.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    assert_eq!(collected, vec![1, 3, 2, 4]);
}

#[test]
fn test_index_iterator_after_contiguous() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 3, 4]
    ).unwrap();

    let transposed = tensor.permute(vec![2, 1, 0]).unwrap();
    let contiguous = transposed.contiguous();

    let style = IterStyle::Cstyle;
    let iter = contiguous.tensor_iterator(style);
    let count = iter.count();

    assert_eq!(count, 24);
}

#[test]
fn test_iterate_flattened_tensor() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6],
        vec![2, 3]
    ).unwrap();

    let flat = tensor.flatten().unwrap();
    let style = IterStyle::Cstyle;
    let mut iter = flat.tensor_iterator(style);

    for i in 1..=6 {
        assert_eq!(iter.next(), Some(i));
    }
    assert_eq!(iter.next(), None);
}

// =============================================================================
// ITERATION ORDER VERIFICATION
// =============================================================================

#[test]
fn test_c_style_order_2d() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![0, 1, 2, 3, 4, 5],
        vec![2, 3]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    assert_eq!(collected, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn test_c_style_order_3d() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 3, 4]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    assert_eq!(collected, (0..24).collect::<Vec<i32>>());
}

#[test]
fn test_f_style_order_2d() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..6).collect(),
        vec![2, 3]
    ).unwrap();

    let style = IterStyle::Fstyle;
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    let expected = vec![0, 3, 1, 4, 2, 5];
    assert_eq!(collected, expected);
}

#[test]
fn test_f_style_order_3d() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..24).collect(),
        vec![2, 3, 4]
    ).unwrap();

    let style = IterStyle::Fstyle;
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    let expected = vec![
        0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21,
        2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23
    ];
    assert_eq!(collected, expected);
}

// =============================================================================
// METADATA INVARIANTS DURING ITERATION
// =============================================================================

#[test]
fn test_iteration_covers_exactly_numel() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        (0..60).collect(),
        vec![3, 4, 5]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    assert_eq!(iter.count(), tensor.numel());
}

#[test]
fn test_iteration_produces_all_valid_indices() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![0; 12],
        vec![3, 4]
    ).unwrap();

    let mut index_iter = tensor.index_iterator(IterStyle::Cstyle);
    let mut seen = vec![vec![0usize; 2]; 12];

    for i in 0..12 {
        seen[i] = index_iter.next().unwrap();
    }

    for idx in &seen {
        assert!(idx[0] < 3);
        assert!(idx[1] < 4);
    }
}

#[test]
fn test_all_indices_unique_during_iteration() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![0; 24],
        vec![2, 3, 4]
    ).unwrap();

    let mut index_iter = tensor.index_iterator(IterStyle::Cstyle);
    let mut seen = std::collections::HashSet::new();

    while let Some(idx) = index_iter.next() {
        let key = (idx[0], idx[1], idx[2]);
        assert!(!seen.contains(&key));
        seen.insert(key);
    }

    assert_eq!(seen.len(), 24);
}

// =============================================================================
// CUSTOM ITERATION ORDER
// =============================================================================

#[test]
fn test_custom_iteration_order_reversed() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let style = IterStyle::Custom(vec![1, 0]);
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    assert_eq!(collected, vec![1, 2, 3, 4]);
}

#[test]
fn test_custom_iteration_order_middle_slowest() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![2, 2, 2]
    ).unwrap();

    let style = IterStyle::Custom(vec![1, 2, 0]);
    let iter = tensor.tensor_iterator(style);
    let collected: Vec<i32> = iter.collect();

    assert_eq!(collected, vec![1, 3, 2, 4, 5, 7, 6, 8]);
}

// =============================================================================
// ITERATOR COLLECTIONS AND ADAPTERS
// =============================================================================

#[test]
fn test_tensor_iterator_sum() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5],
        vec![5]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    let sum: i32 = iter.sum();

    assert_eq!(sum, 15);
}

#[test]
fn test_tensor_iterator_fold() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4],
        vec![2, 2]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);
    let product: i32 = iter.fold(1, |acc, x| acc * x);

    assert_eq!(product, 24);
}

#[test]
fn test_tensor_iterator_enumerate() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![10, 20, 30],
        vec![3]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let iter = tensor.tensor_iterator(style);

    let indexed: Vec<(usize, i32)> = iter.enumerate().map(|(i, v)| (i, v)).collect();

    assert_eq!(indexed, vec![(0, 10), (1, 20), (2, 30)]);
}

#[test]
fn test_tensor_iterator_find() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5],
        vec![5]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);
    let found = iter.find(|&x| x > 3);

    assert_eq!(found, Some(4));
}

#[test]
fn test_tensor_iterator_any() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![1, 2, 3, 4, 5],
        vec![5]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);
    assert!(iter.any(|x| x == 3));
}

#[test]
fn test_tensor_iterator_all() {
    let tensor: Tensor<i32> = Tensor::from_vector(
        vec![2, 4, 6, 8],
        vec![4]
    ).unwrap();

    let style = IterStyle::Cstyle;
    let mut iter = tensor.tensor_iterator(style);
    assert!(iter.all(|x| x % 2 == 0));
}