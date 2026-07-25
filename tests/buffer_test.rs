
use pulsars::buffer::Buffer;

#[test]
fn test_buffer_construction(){
    let buffer: Buffer<i16> = Buffer::new();

    assert_eq!(buffer.into_vec(), vec![])
}

#[test]
fn test_buffer_construction_with_capacity(){
    let buffer: Buffer<i16> = Buffer::with_capacity(5);

    assert_eq!(buffer.capacity(), 5)
}

#[test]
fn test_buffer_length_and_capacity(){
    let mut buffer: Buffer<i16> = Buffer::with_capacity(5);

    buffer.push(1);
    buffer.push(2);
    buffer.push(3);

    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.capacity(), 5);
}

#[test]
fn test_buffer_isempty_when_empty(){
    let buffer: Buffer<i16> = Buffer::with_capacity(5);

    assert_eq!(buffer.is_empty(), true)
}

#[test]
fn test_buffer_isempty_when_not_empty(){
    let mut buffer: Buffer<i16> = Buffer::new();
    buffer.push(1);

    assert_eq!(buffer.is_empty(), false)
}

#[test]
fn test_buffer_pushing_data(){
    let mut buffer: Buffer<i16> = Buffer::with_capacity(5);

    buffer.push(1);

    assert_eq!(buffer.into_vec(), vec![1])
}

#[test]
fn test_buffer_slice(){
    let mut buffer: Buffer<i16> = Buffer::with_capacity(5);

    buffer.push(1);

    let slice = buffer.as_slice();

    assert_eq!(slice, [1]);
}

#[test]
fn test_buffer_mutable_slice(){
    let mut buffer: Buffer<i16> = Buffer::with_capacity(5);

    buffer.push(1);
    buffer.push(2);

    let m_slice = buffer.as_mut_slice();

    m_slice[1] = 23;

    assert_eq!(m_slice[1], 23)
}

#[test]
fn test_buffer_construction_from_vector(){
    let buffer: Buffer<i16> = Buffer::from_vec(vec![1, 2, 3, 4]);

    assert_eq!(buffer.as_slice(), &[1, 2, 3, 4])
}

