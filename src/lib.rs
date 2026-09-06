pub mod buffer;
pub mod cursor;
pub mod dtype;
pub mod error;
pub mod iterator;
pub mod shape;
pub mod stride;
pub mod tensor;
pub mod prelude;
pub mod random;
pub mod slicing;

// function signature helper
#[macro_export]
macro_rules! function_signature {
    () => {{
        fn f() {}
        fn fxname<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }

        let fn_name = fxname(f);
        fn_name.strip_suffix("::f").unwrap_or(fn_name)
    }};
}
