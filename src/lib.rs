//! # Pulsrs
//!
//! Pulsrs is a long-term exploration into how modern machine-learning
//! computation systems work underneath the abstractions we use every day.
//!
//! The project started as a way to learn the mathematics behind machine
//! learning and understand how tensor computation is implemented underneath
//! frameworks such as NumPy and PyTorch.
//!
//! What started as a learning project has grown into a much larger vision:
//! to eventually build a full-fledged ecosystem for machine-learning model
//! development and training.
//!
//! ## Philosophy
//!
//! Pulsrs is built around three core values:
//!
//! - **Correctness**
//! - **Speed**
//! - **Efficiency**
//!
//! Pulsrs is currently nowhere near achieving all three at a production level.
//! The project is still in its foundational stages, and there is a long road
//! ahead. However, these principles influence the architecture from the very
//! beginning.
//!
//! The goal is not simply to make tensor operations work, but to understand
//! how the underlying abstractions can eventually support efficient,
//! high-performance computation.
//!
//! ## Why Pulsrs?
//!
//! You might ask:
//!
//! > "Why build another tensor library? They already exist."
//!
//! And that's correct.
//!
//! Mature and highly optimized systems already exist. Pulsrs is not intended
//! to replace them today.
//!
//! I wanted to build one myself.
//!
//! Building Pulsrs from the ground up provides an opportunity to understand
//! what happens underneath familiar machine-learning APIs:
//!
//! - How are tensors represented in memory?
//! - How can tensor views avoid copying data?
//! - How do shapes and strides describe a tensor?
//! - How does slicing interact with arbitrary memory layouts?
//! - How does broadcasting actually work?
//! - How can these abstractions eventually support automatic differentiation?
//!
//! Pulsrs is therefore as much a learning project as it is a software project.
//!
//! ## The Name
//!
//! Pulsrs takes its name from **pulsars** — rapidly rotating celestial objects
//! that emit extraordinarily regular pulses of radiation.
//!
//! The inspiration is simple:
//!
//! > I wanted to build a tensor computation engine that could eventually be
//! > blazingly fast.
//!
//! Whether it actually becomes blazingly fast is a problem for future me.
//!
//! ## Current Capabilities
//!
//! Pulsrs is currently in its foundational tensor-computation stage.
//!
//! Current capabilities include:
//!
//! - Tensors
//! - Tensor iterators
//! - Tensor mappings
//! - Basic tensor operations
//! - Tensor slicing
//! - Broadcasting
//! - Zero-copy tensor views
//!
//! Although these capabilities are fundamental, they establish several
//! important design principles for the system.
//!
//! ### Zero-Copy Views
//!
//! Tensor views do not require copying the underlying data.
//!
//! Different tensor representations can therefore operate over the same
//! underlying memory while interpreting it through different shapes and
//! strides.
//!
//! ### Amorphous Memory
//!
//! Pulsrs separates the concept of memory from the shape of a tensor.
//!
//! The underlying buffer is essentially a collection of values. A tensor
//! interprets that buffer through its shape and stride metadata rather than
//! requiring the memory itself to conform to a particular dimensional
//! structure.
//!
//! This separation provides the foundation for operations such as reshaping,
//! permutation and views.
//!
//! ### Slicing
//!
//! Pulsrs supports tensor slicing, including negative steps.
//!
//! Slicing is represented through tensor metadata and strides, allowing
//! slices to operate as views over existing memory rather than requiring
//! unnecessary data copies.
//!
//! ### Broadcasting
//!
//! Pulsrs supports broadcasting through tensor views and stride manipulation.
//!
//! Broadcast dimensions can be represented without physically replicating
//! their underlying values, allowing logically expanded tensors to operate
//! without unnecessary memory allocation.
//!
//! ### Iterator-Aware Memory Layout
//!
//! Tensor iterators respond to the tensor's underlying memory layout.
//!
//! Iteration is therefore not based solely on the tensor's logical shape.
//! The iterator also respects the mapping between tensor indices and the
//! underlying memory.
//!
//! This becomes increasingly important as Pulsrs supports non-contiguous
//! tensors and more complex tensor transformations.
//!
//! ## What's Next?
//!
//! Pulsrs is still building its foundations.
//!
//! Before moving toward higher-level machine-learning functionality, the core
//! tensor infrastructure needs to become robust, well-tested and extensible.
//!
//! One of the next major milestones is **automatic differentiation**.
//!
//! Autograd will represent an important transition for Pulsrs: moving from a
//! tensor calculation engine toward a system capable of representing and
//! differentiating computational graphs.
//!
//! The long-term direction is expected to grow along the lines of:
//!
//! ```text
//! Tensor Computation
//!         ↓
//!      Autograd
//!         ↓
//! Neural Network Primitives
//!         ↓
//!  Model Development
//!         ↓
//!       Training
//!         ↓
//!    Optimization
//!         ↓
//!     Execution
//! ```
//!
//! The exact architecture will evolve as the project grows.
//!
//! ## The Long-Term Dream
//!
//! Pulsrs is ambitious.
//!
//! The ultimate dream is to build an ecosystem capable of competing with
//! established machine-learning frameworks — perhaps even PyTorch one day.
//!
//! That's obviously a ridiculous goal for a project that is currently still
//! working on its tensor foundations.
//!
//! But that's also why it's worth pursuing.
//!
//! > Dream big to achieve big.
//! >
//! > Dream on.
//!
//! ## Project Status
//!
//! Pulsrs is an active work in progress.
//!
//! The current focus is on establishing a correct and flexible tensor
//! foundation before moving toward higher-level machine-learning systems.
//!
//! Performance optimization is a long-term objective, not a claim about the
//! current implementation.

pub mod buffer;
pub mod cursor;
pub mod dtype;
pub mod error;
pub mod iterator;
pub mod kernels;
pub mod prelude;
pub mod random;
pub mod shape;
pub(crate) mod simd;
pub mod slicing;
pub mod stride;
pub mod tensor;

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
