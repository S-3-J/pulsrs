
use pulsars::shape::shape::Shape;

#[test]
fn test_shape_rank() {
    let shape = Shape::new(vec![3,4,5]);
    let shape_sc = Shape::scalar();
    
    assert_eq!(shape.rank(), 3);
    assert_eq!(shape_sc.rank(), 0);
}

#[test]
fn test_shape_ndims() {
    let shape = Shape::new(vec![3,4,5]);
    let shape_sc = Shape::scalar();
    
    assert_eq!(shape.ndim(), 3);
    assert_eq!(shape_sc.ndim(), 0);
}

#[test]
fn test_shape_numel() {
    let shape = Shape::new(vec![3,4,5]);
    let shape_sc = Shape::scalar();
    let shape_z = Shape::new(vec![3,0,5]);
    
    assert_eq!(shape.numel(), 3*4*5);
    assert_eq!(shape_sc.numel(), 1);
    assert_eq!(shape_z.numel(), 0);
}

#[test]
fn test_shape_dims() {
    let shape = Shape::new(vec![3,4,5]);
    let shape_sc = Shape::scalar();
    
    assert_eq!(shape.dims(), &[3,4,5], "check 1");
    assert_eq!(shape_sc.dims(), &[], "check 2")
}

#[test]
fn test_shape_is_scalar() {
    let shape = Shape::new(vec![3,4,5]);
    let shape_sc = Shape::scalar();
    
    assert!(!shape.is_scalar(), "check 1");
    assert!(shape_sc.is_scalar(), "check 2");
}

#[test]
fn test_shape_is_empty() {
    let shape = Shape::new(vec![3,4,5]);
    let shape_emt = Shape::new(vec![1,0,4]);
    let shape_sc = Shape::scalar();
    
    assert!(!shape.is_empty(), "check 1");
    assert!(!shape_sc.is_empty(), "check 2");
    assert!(shape_emt.is_empty(), "check 3");
}

#[test]
fn test_shape_from_a_vector() {
    let shape : Shape = vec![3,4,5].into();
    let shape_2 : Shape = Shape::new(vec![3,4,5]);
    
    assert_eq!(shape, shape_2);
}

#[test]
fn test_shape_default() {
    let shape : Shape = Shape::default();
    let shape_2 : Shape = Shape::scalar();
    assert_eq!(shape, shape_2);
}

#[test]
fn test_shape_get() {
    let shape : Shape = vec![3,4,5].into();
    
    assert_eq!(shape.get(1), Some(4));
    assert_eq!(shape.get(3), None);
}

#[test]
fn test_shape_iter() {
    let shape : Shape = vec![3,4,5].into();

    let d : Vec<usize> = vec![3,4,5];

    assert_eq!(shape.iter().len(), d.iter().len());

}
