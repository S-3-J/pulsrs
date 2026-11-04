//! # Buffer
//! Buffer, short for Memory buffer, holds data in contiguous allocation. It serves as the underlying
//! data container for a Tensor.
//! It is a data structure that represents a 1-D array.
//!
//! ---
//!
//! ## Structure
//!
//! Buffer is just a wrapper over a rust Vector.
//! ```text
//!     pub struct Buffer<T> {
//!         data: Vec<T>
//!     }
//! ```
//! This makes buffer pretty straight-forward.
//!
//! ## Access
//! A Buffer, supports `Index` and `IndexMut` traits, and hence can be access via rust's standard indexing norms.
//!

#[doc(hidden)]
pub mod buffer_methods;
pub mod buffer_types;

#[derive(Debug)]
pub struct Buffer<T> {
    data: Vec<T>,
}
