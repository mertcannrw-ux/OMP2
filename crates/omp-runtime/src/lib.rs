//! omp-runtime crate.
//!
//! Enforces the trusted host / sandbox runtime boundary and provides the unified
//! killable Job primitive, bounded stream accumulation, artifact metadata management,
//! and isolated workspace view descriptors.

pub mod artifact;
pub mod job;
pub mod stream;
pub mod workspace;

pub use artifact::{
    Artifact, ArtifactMetadata, ArtifactOrigin, ArtifactRetention, ArtifactScope, ArtifactStore,
};
pub use job::{Job, JobError, JobKind, JobSignal, JobTerminationReason};
pub use omp_types::{
    LimitPolicy, SandboxCapability, SandboxEvent, SandboxExitStatus, SandboxRequest, SandboxUsage,
    TruncationDiag, WorkspaceViewId,
};
pub use stream::{BoundedStream, BoundedStreamAccumulator, StreamChunk, StreamKind};
pub use workspace::{
    CowBackend, DiffKind, FileDiffEntry, IsolationMode, WorkspaceDiff, WorkspaceView,
};
pub mod process;
#[cfg(windows)]
pub mod windows_sandbox;
pub use process::RestrictedProcess;
