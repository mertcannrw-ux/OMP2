//! omp-inference: provider adapters, model taxonomy/compiler, compatibility resolution,
//! capability policy, output repair, token usage, and speculative compaction.

pub mod capability;
pub mod compaction;
pub mod compat;
pub mod corrective;
pub mod local_model;
pub mod model_taxonomy;
pub mod provider;
pub mod repair;
pub mod request;
pub mod tool_force;

pub use capability::*;
pub use compaction::*;
pub use compat::*;
pub use corrective::*;
pub use local_model::*;
pub use model_taxonomy::*;
pub use provider::*;
pub use repair::*;
pub use request::*;
pub use tool_force::*;
