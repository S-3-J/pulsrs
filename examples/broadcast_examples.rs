use pulsars::prelude::*;

fn main() {
    // ============================================================
    // 1. Equal shapes
    // ============================================================

    let a = Tensor::<i8>::arange(1, 6, 1)
        .reshape(vec![2, 3])
        .unwrap();

    let b = Tensor::<i8>::arange(1, 6, 1)
        .reshape(vec![2, 3])
        .unwrap();

    println!("TEST 1: [2, 3] + [2, 3]");
    println!("{:?}\n", a.is_broadcastible_for(b.shape().dims()));


    // ============================================================
    // 2. [3] -> [2, 3]
    // ============================================================

    let a = Tensor::<i8>::arange(1, 3, 1)
        .reshape(vec![3])
        .unwrap();

    let b = Tensor::<i8>::arange(1, 6, 1)
        .reshape(vec![2, 3])
        .unwrap();

    println!("TEST 2: [3] + [2, 3]");
    println!("{:?}", a.is_broadcastible_for(b.shape().dims()));

    let c = a.broadcast(&[2, 3]);

    println!("Broadcasted tensor:");
    println!("{}", c);
    println!("Shape: {:?}", c.shape().dims());
    println!("Stride: {:?}\n", c.strides());


    // ============================================================
    // 3. [1, 3] -> [2, 3]
    // ============================================================

    let a = Tensor::<i8>::arange(1, 3, 1)
        .reshape(vec![1, 3])
        .unwrap();

    let b = Tensor::<i8>::arange(1, 6, 1)
        .reshape(vec![2, 3])
        .unwrap();

    println!("TEST 3: [1, 3] + [2, 3]");
    println!("{:?}", a.is_broadcastible_for(b.shape().dims()));

    let c = a.broadcast(&[2, 3]);

    println!("Shape: {:?}", c.shape().dims());
    println!("Stride: {:?}\n", c.strides());


    // ============================================================
    // 4. Middle-dimension broadcasting
    // [2, 1, 3] -> [2, 4, 3]
    // ============================================================

    let a = Tensor::<i8>::arange(1, 6, 1)
        .reshape(vec![2, 1, 3])
        .unwrap();

    let b = Tensor::<i8>::arange(1, 24, 1)
        .reshape(vec![2, 4, 3])
        .unwrap();

    println!("TEST 4: [2, 1, 3] + [2, 4, 3]");
    println!("{:?}", a.is_broadcastible_for(b.shape().dims()));

    let c = a.broadcast(&[2, 4, 3]);

    println!("{}", c);
    println!("Shape: {:?}", c.shape().dims());
    println!("Stride: {:?}\n", c.strides());


    // ============================================================
    // 5. New leading dimension
    // [5, 2, 3] + [2, 3]
    // ============================================================

    let a = Tensor::<i8>::arange(1, 30, 1)
        .reshape(vec![5, 2, 3])
        .unwrap();

    let b = Tensor::<i8>::arange(1, 6, 1)
        .reshape(vec![2, 3])
        .unwrap();

    println!("TEST 5: [5, 2, 3] + [2, 3]");
    println!("{:?}", a.is_broadcastible_for(b.shape().dims()));

    println!("Reverse:");
    println!("{:?}\n", b.is_broadcastible_for(a.shape().dims()));


    // ============================================================
    // 6. Incompatible shapes
    // ============================================================

    let a = Tensor::<i8>::arange(1, 6, 1)
        .reshape(vec![2, 3])
        .unwrap();

    let b = Tensor::<i8>::arange(1, 8, 1)
        .reshape(vec![2, 4])
        .unwrap();

    println!("TEST 6: [2, 3] + [2, 4]");
    println!("{:?}\n", a.is_broadcastible_for(b.shape().dims()));


    // ============================================================
    // 7. Your original complex case
    // [5, 2, 3]
    //      ↓ None
    // [5, 1, 2, 3]
    //      ↓ broadcast
    // [2, 5, 2, 2, 3]
    // ============================================================

    let t2 = Tensor::<i8>::arange(1, 30, 1)
        .reshape(vec![5, 2, 3])
        .unwrap();

    let sliced = t2
        .slice(vec![
            (..).iosr(),
            None,
            (..).iosr(),
            (..).iosr(),
        ])
        .unwrap();

    println!("TEST 7: Complex broadcast");
    println!("Sliced shape: {:?}", sliced.shape().dims());
    println!("Sliced stride: {:?}", sliced.strides());

    let t3 = sliced.broadcast(&[2, 5, 2, 2, 3]);

    println!("Broadcasted shape: {:?}", t3.shape().dims());
    println!("Broadcasted stride: {:?}", t3.strides());
    println!("{}", t3);
}