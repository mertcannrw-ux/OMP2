//! omp-render crate.
//!
//! Provides RichText and streaming Out sinks, typed component tree with semantic
//! registry, and transcript block lifecycle with ElasticSlots protocol invariants.

pub mod component;
pub mod debug;
pub mod out;
pub mod richtext;
pub mod semantic;
pub mod terminal;
pub mod transcript;

pub use component::*;
pub use debug::*;
pub use out::*;
pub use richtext::*;
pub use semantic::*;
pub use terminal::*;
pub use transcript::*;
