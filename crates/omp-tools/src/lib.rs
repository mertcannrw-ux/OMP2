//! omp-tools crate.
//!
//! Provides the stable, deep tool surface for Harness Oh my Pi 2 (omp2).
//! Implements ToolDefinition, ToolCall, HostGateway, and the permanent core roster:
//! Read, Bash, Write, Edit, Eval, Agent, and AutoQA, plus the dyn discovery protocol.

pub mod agent;
pub mod autoqa;
pub mod bash;
pub mod definition;
pub mod dyn_discovery;
pub mod edit;
pub mod eval;
pub mod host;
pub mod read;
mod resource;
pub mod roster;
mod ssh;
pub mod write;

pub use agent::*;
pub use autoqa::*;
pub use bash::*;
pub use definition::*;
pub use dyn_discovery::*;
pub use edit::*;
pub use eval::*;
pub use host::*;
pub use read::*;
pub use roster::*;
pub use write::*;
