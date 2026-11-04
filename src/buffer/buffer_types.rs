/*
use std::{ops::Deref, sync::{Arc, Mutex}};

use super::Buffer;

pub struct Immutable<T> {
    buffer: Arc<Buffer<T>>
}

pub struct Mutable<T> {
    buffer: Arc<Mutex<Buffer<T>>>
}

pub struct SharedMutable<T> {
    buffer: Arc<Mutex<Buffer<T>>>
}

impl<T> Deref for Immutable<T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T> Deref for Mutable<T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
} */
