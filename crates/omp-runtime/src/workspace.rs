use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use omp_types::WorkspaceViewId;
use omp_types::{SessionId, StructuredError};

/// OS-locked exclusive mutable lease for an isolated workspace view.
///
/// Uses OS file locking (`file.try_lock()`) on a stable lease file located outside
/// the workspace view in the trusted `leases/` directory.
///
/// Dropping or releasing the lease closes the file handle, which immediately releases
/// the OS lock. The file is never unlinked to prevent inode/path race conditions and
/// eliminate stale marker deadlocks after a process crash.
#[derive(Debug)]
pub struct WorkspaceLease {
    view_id: WorkspaceViewId,
    lease_path: PathBuf,
    /// Never read; held so dropping the lease releases the OS file lock.
    #[allow(dead_code)]
    file: File,
}

impl WorkspaceLease {
    pub fn view_id(&self) -> &WorkspaceViewId {
        &self.view_id
    }

    pub fn lease_path(&self) -> &Path {
        &self.lease_path
    }

    /// Explicitly release the exclusive lease by closing the OS file handle.
    /// Never unlinks the file to prevent inode race conditions.
    pub fn release(self) {
        drop(self);
    }
}

impl Drop for WorkspaceLease {
    fn drop(&mut self) {
        // Closing the File automatically releases the OS-level lock.
        // The lease file is intentionally left on disk to prevent inode races.
    }
}

/// Baseline file state captured at allocation time for immutable diff computation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FileBaseline {
    pub size: u64,
    pub sha256: String,
}

/// Supported copy-on-write filesystem backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CowBackend {
    Apfs,
    Btrfs,
    Zfs,
    Overlayfs,
    ProjFs,
    Reflink,
}

/// Workspace isolation mode used by the host when spawning child tasks or subagents.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IsolationMode {
    /// Tracked-file worktree (e.g. Git worktree).
    Worktree,
    /// Whole-workspace copy-on-write view.
    CopyOnWrite { backend: CowBackend },
    /// Physical directory copy fallback when CoW or worktree cannot be allocated.
    CopyFallback,
    /// Direct host execution within the workspace directory itself.
    Direct,
}

/// Descriptor of an isolated workspace view allocated to a job or subagent.
/// Contains durable baseline file state to guarantee diffing against immutable allocation baseline
/// rather than the mutable live parent.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceView {
    pub view_id: WorkspaceViewId,
    pub session_id: SessionId,
    pub base_path: PathBuf,
    pub isolated_path: PathBuf,
    pub isolation_mode: IsolationMode,
    pub read_only: bool,
    pub allowed_globs: Vec<String>,
    pub denied_globs: Vec<String>,
    #[serde(default)]
    pub baseline_files: HashMap<PathBuf, FileBaseline>,
    #[serde(default)]
    pub baseline_directories: std::collections::BTreeSet<PathBuf>,
}

impl WorkspaceView {
    pub fn new(
        view_id: WorkspaceViewId,
        session_id: SessionId,
        base_path: impl AsRef<Path>,
        isolated_path: impl AsRef<Path>,
        isolation_mode: IsolationMode,
    ) -> Self {
        Self {
            view_id,
            session_id,
            base_path: base_path.as_ref().to_path_buf(),
            isolated_path: isolated_path.as_ref().to_path_buf(),
            isolation_mode,
            read_only: false,
            allowed_globs: Vec::new(),
            denied_globs: Vec::new(),
            baseline_files: HashMap::new(),
            baseline_directories: Default::default(),
        }
    }

    /// Allocate a direct workspace view operating directly in the workspace directory.
    pub fn direct(base: impl AsRef<Path>, session_id: SessionId) -> Result<Self, StructuredError> {
        let base_raw = base.as_ref();
        if !base_raw.exists() {
            return Err(StructuredError::new(
                "base_not_found",
                format!("Base workspace path does not exist: {}", base_raw.display()),
                false,
            ));
        }
        let base_canon = base_raw.canonicalize().map_err(|e| {
            StructuredError::new(
                "canonicalize_error",
                format!(
                    "Failed to canonicalize base path {}: {}",
                    base_raw.display(),
                    e
                ),
                false,
            )
        })?;
        let view_id = WorkspaceViewId::mint();
        Ok(Self {
            view_id,
            session_id,
            base_path: base_canon.clone(),
            isolated_path: base_canon,
            isolation_mode: IsolationMode::Direct,
            read_only: false,
            allowed_globs: Vec::new(),
            denied_globs: Vec::new(),
            baseline_files: HashMap::new(),
            baseline_directories: Default::default(),
        })
    }

    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    pub fn with_allowed_glob(mut self, glob: impl Into<String>) -> Self {
        self.allowed_globs.push(glob.into());
        self
    }

    pub fn with_denied_glob(mut self, glob: impl Into<String>) -> Self {
        self.denied_globs.push(glob.into());
        self
    }

    /// Path to the exclusive lease lock file outside the isolated view directory.
    pub fn lease_path(&self) -> PathBuf {
        let parent = if self.isolation_mode == IsolationMode::Direct {
            self.base_path.join(".omp")
        } else {
            self.isolated_path
                .parent()
                .unwrap_or(&self.isolated_path)
                .to_path_buf()
        };
        parent
            .join("leases")
            .join(format!("{}.lease", self.view_id.as_str()))
    }

    /// Acquire exclusive OS-locked mutable lease for this view using OS file locking.
    ///
    /// - Uses `OpenOptions::create(true)` to create or reuse the stable lease path in `leases/`.
    /// - Locks the file via `file.try_lock()` (RFC 3600 / Rust 1.97 standard library file locking).
    /// - Never unlinks the file on drop, preventing inode race conditions and allowing crash recovery.
    pub fn acquire(&self) -> Result<WorkspaceLease, StructuredError> {
        let lease_path = self.lease_path();
        if let Some(parent) = lease_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                StructuredError::new(
                    "lease_dir_error",
                    format!(
                        "Failed to create leases directory {}: {}",
                        parent.display(),
                        e
                    ),
                    false,
                )
            })?;
        }

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lease_path)
            .map_err(|e| {
                StructuredError::new(
                    "lease_open_error",
                    format!("Failed to open lease file {}: {}", lease_path.display(), e),
                    false,
                )
            })?;

        file.try_lock().map_err(|e| {
            StructuredError::new(
                "view_already_leased",
                format!(
                    "Workspace view {} is currently locked by another active job or process: {}",
                    self.view_id.as_str(),
                    e
                ),
                false,
            )
        })?;

        Ok(WorkspaceLease {
            view_id: self.view_id.clone(),
            lease_path,
            file,
        })
    }

    /// Allocate a real unique isolated workspace copy outside the parent source.
    ///
    /// Default allocation uses safe pure-Rust physical copy fallback (`IsolationMode::CopyFallback`)
    /// and invokes NO host Git commands, ensuring absolute immunity against hostile repository exploits
    /// (malicious hooks, arbitrary filter execution, or `core.sshCommand`).
    pub fn allocate(
        base: impl AsRef<Path>,
        views_root: impl AsRef<Path>,
        session_id: SessionId,
    ) -> Result<Self, StructuredError> {
        Self::allocate_internal(base, views_root, session_id, false)
    }

    /// Explicitly trusted worktree allocation policy.
    /// Only use when the repository is verified trusted by explicit host policy.
    pub fn allocate_worktree_trusted(
        base: impl AsRef<Path>,
        views_root: impl AsRef<Path>,
        session_id: SessionId,
    ) -> Result<Self, StructuredError> {
        Self::allocate_internal(base, views_root, session_id, true)
    }

    /// Determines the safe views_root outside the parent workspace.
    /// Defaults to `%TEMP%/omp2-workspaces` unless the workspace contains temp_dir (or vice-versa),
    /// in which case it allocates in a sibling `.omp2-views` outside the workspace.
    pub fn default_views_root(workspace: &Path) -> PathBuf {
        let temp = std::env::temp_dir().join("omp2-workspaces");
        let ws_canon = workspace
            .canonicalize()
            .unwrap_or_else(|_| workspace.to_path_buf());
        let temp_canon = temp.canonicalize().unwrap_or_else(|_| temp.clone());
        if temp_canon.starts_with(&ws_canon)
            || ws_canon.starts_with(&temp_canon)
            || temp_canon == ws_canon
        {
            if let Some(parent) = ws_canon.parent() {
                parent.join(".omp2-views")
            } else {
                std::env::temp_dir().join("omp2-workspaces")
            }
        } else {
            temp
        }
    }

    fn allocate_internal(
        base: impl AsRef<Path>,
        views_root: impl AsRef<Path>,
        session_id: SessionId,
        trusted_worktree: bool,
    ) -> Result<Self, StructuredError> {
        let base_raw = base.as_ref();
        let views_raw = views_root.as_ref();

        if !base_raw.exists() {
            return Err(StructuredError::new(
                "base_not_found",
                format!("Base workspace path does not exist: {}", base_raw.display()),
                false,
            ));
        }

        let base_canon = base_raw.canonicalize().map_err(|e| {
            StructuredError::new(
                "canonicalize_error",
                format!(
                    "Failed to canonicalize base path {}: {}",
                    base_raw.display(),
                    e
                ),
                false,
            )
        })?;

        fs::create_dir_all(views_raw).map_err(|e| {
            StructuredError::new(
                "views_root_error",
                format!("Failed to create views_root {}: {}", views_raw.display(), e),
                false,
            )
        })?;

        let views_canon = views_raw.canonicalize().map_err(|e| {
            StructuredError::new(
                "canonicalize_error",
                format!(
                    "Failed to canonicalize views_root {}: {}",
                    views_raw.display(),
                    e
                ),
                false,
            )
        })?;

        // Invariant: Cannot allocate alias of parent, views must stay outside parent source
        if views_canon == base_canon {
            return Err(StructuredError::new(
                "alias_of_parent",
                "views_root cannot be identical to the base workspace (cannot allocate alias of parent)",
                false,
            ));
        }

        if views_canon.starts_with(&base_canon) {
            return Err(StructuredError::new(
                "views_inside_parent",
                format!(
                    "views_root ({}) must be outside parent source ({})",
                    views_canon.display(),
                    base_canon.display()
                ),
                false,
            ));
        }

        if base_canon.starts_with(&views_canon) {
            return Err(StructuredError::new(
                "parent_inside_views",
                format!(
                    "Base workspace ({}) cannot be inside views_root ({})",
                    base_canon.display(),
                    views_canon.display()
                ),
                false,
            ));
        }

        // Capture durable baseline files at allocation time from base workspace
        let baseline_files = scan_workspace_files(&base_canon).map_err(|e| {
            StructuredError::new(
                "baseline_scan_error",
                format!("Failed to scan baseline files in base workspace: {}", e),
                false,
            )
        })?;
        let baseline_directories = scan_directories(&base_canon).map_err(|error| {
            StructuredError::new("baseline_scan_error", error.to_string(), false)
        })?;

        let view_id = WorkspaceViewId::mint();
        let isolated_path = views_canon.join(view_id.as_str());

        if isolated_path.exists() {
            return Err(StructuredError::new(
                "isolated_path_exists",
                format!("Isolated path already exists: {}", isolated_path.display()),
                false,
            ));
        }

        let isolation_mode = if trusted_worktree && base_canon.join(".git").exists() {
            let worktree_res = Command::new("git")
                .args([
                    "-c",
                    "core.hooksPath=NUL",
                    "-c",
                    "core.fsmonitor=false",
                    "-c",
                    "filter.lfs.smudge=",
                    "-c",
                    "filter.lfs.clean=",
                    "-c",
                    "filter.lfs.process=",
                    "-c",
                    "filter.lfs.required=false",
                    "worktree",
                    "add",
                    "--detach",
                    isolated_path.to_str().unwrap_or_default(),
                    "HEAD",
                ])
                .current_dir(&base_canon)
                .output();

            match worktree_res {
                Ok(out) if out.status.success() => {
                    copy_untracked_files(&base_canon, &isolated_path).map_err(|e| {
                        StructuredError::new(
                            "untracked_copy_failed",
                            format!("Failed copying untracked files into worktree: {}", e),
                            false,
                        )
                    })?;
                    IsolationMode::Worktree
                }
                _ => {
                    copy_directory_safe(&base_canon, &isolated_path, &base_canon).map_err(|e| {
                        StructuredError::new(
                            "workspace_copy_failed",
                            format!("Failed copying workspace into isolated view: {}", e),
                            false,
                        )
                    })?;
                    IsolationMode::CopyFallback
                }
            }
        } else {
            // Default hostile mode: purely safe copy fallback, zero host git commands invoked
            copy_directory_safe(&base_canon, &isolated_path, &base_canon).map_err(|e| {
                StructuredError::new(
                    "workspace_copy_failed",
                    format!("Failed copying workspace into isolated view: {}", e),
                    false,
                )
            })?;
            IsolationMode::CopyFallback
        };

        Ok(Self {
            view_id,
            session_id,
            base_path: base_canon,
            isolated_path,
            isolation_mode,
            read_only: false,
            allowed_globs: Vec::new(),
            denied_globs: Vec::new(),
            baseline_files,
            baseline_directories,
        })
    }

    /// Compute structured diff comparing the isolated workspace view against the
    pub fn compute_diff(&self) -> Result<WorkspaceDiff, StructuredError> {
        if self.isolation_mode == IsolationMode::Direct {
            return Ok(WorkspaceDiff::empty(self.view_id.clone()));
        }
        let isolated_files = scan_workspace_files(&self.isolated_path).map_err(|e| {
            StructuredError::new(
                "diff_scan_error",
                format!(
                    "Failed to scan isolated workspace {}: {}",
                    self.isolated_path.display(),
                    e
                ),
                false,
            )
        })?;

        let mut entries = Vec::new();
        let mut matched_baseline_keys = HashSet::new();
        let mut created_entries = Vec::new();
        let mut deleted_entries = Vec::new();

        // Compare files in isolated view against immutable baseline
        for (rel_path, iso_info) in &isolated_files {
            match self.baseline_files.get(rel_path) {
                Some(base_info) => {
                    matched_baseline_keys.insert(rel_path.clone());
                    if iso_info.sha256 != base_info.sha256 {
                        let bytes_changed = if iso_info.size >= base_info.size {
                            (iso_info.size - base_info.size) as usize
                        } else {
                            (base_info.size - iso_info.size) as usize
                        };

                        entries.push(FileDiffEntry {
                            path: rel_path.clone(),
                            kind: DiffKind::Modified,
                            bytes_changed,
                            before_bytes: Some(base_info.size),
                            after_bytes: Some(iso_info.size),
                            before_hash: Some(base_info.sha256.clone()),
                            after_hash: Some(iso_info.sha256.clone()),
                        });
                    }
                }
                None => {
                    created_entries.push((rel_path.clone(), iso_info.clone()));
                }
            }
        }

        // Detect deleted files present in baseline but missing in isolated view
        for (rel_path, base_info) in &self.baseline_files {
            if !matched_baseline_keys.contains(rel_path) {
                deleted_entries.push((rel_path.clone(), base_info.clone()));
            }
        }

        // Rename detection: correlate deleted and created files with identical SHA256 and size
        let mut paired_created = HashSet::new();
        let mut paired_deleted = HashSet::new();

        for (del_idx, (del_path, del_info)) in deleted_entries.iter().enumerate() {
            for (cr_idx, (cr_path, cr_info)) in created_entries.iter().enumerate() {
                if !paired_created.contains(&cr_idx)
                    && del_info.sha256 == cr_info.sha256
                    && del_info.size == cr_info.size
                {
                    paired_deleted.insert(del_idx);
                    paired_created.insert(cr_idx);

                    entries.push(FileDiffEntry {
                        path: cr_path.clone(),
                        kind: DiffKind::Renamed {
                            from: del_path.clone(),
                        },
                        bytes_changed: 0,
                        before_bytes: Some(del_info.size),
                        after_bytes: Some(cr_info.size),
                        before_hash: Some(del_info.sha256.clone()),
                        after_hash: Some(cr_info.sha256.clone()),
                    });
                    break;
                }
            }
        }

        // Add remaining created files
        for (idx, (cr_path, cr_info)) in created_entries.into_iter().enumerate() {
            if !paired_created.contains(&idx) {
                entries.push(FileDiffEntry {
                    path: cr_path,
                    kind: DiffKind::Created,
                    bytes_changed: cr_info.size as usize,
                    before_bytes: None,
                    after_bytes: Some(cr_info.size),
                    before_hash: None,
                    after_hash: Some(cr_info.sha256),
                });
            }
        }

        // Add remaining deleted files
        for (idx, (del_path, del_info)) in deleted_entries.into_iter().enumerate() {
            if !paired_deleted.contains(&idx) {
                entries.push(FileDiffEntry {
                    path: del_path,
                    kind: DiffKind::Deleted,
                    bytes_changed: del_info.size as usize,
                    before_bytes: Some(del_info.size),
                    after_bytes: None,
                    before_hash: Some(del_info.sha256),
                    after_hash: None,
                });
            }
        }

        // Deterministic sorting by relative path
        entries.sort_by(|a, b| a.path.cmp(&b.path));

        let summary = if entries.is_empty() {
            "no changes".to_string()
        } else {
            let mut created_c = 0;
            let mut modified_c = 0;
            let mut deleted_c = 0;
            let mut renamed_c = 0;

            for entry in &entries {
                match &entry.kind {
                    DiffKind::Created => created_c += 1,
                    DiffKind::Modified => modified_c += 1,
                    DiffKind::Deleted => deleted_c += 1,
                    DiffKind::Renamed { .. } => renamed_c += 1,
                }
            }

            let mut parts = Vec::new();
            if created_c > 0 {
                parts.push(format!("{} created", created_c));
            }
            if modified_c > 0 {
                parts.push(format!("{} modified", modified_c));
            }
            if deleted_c > 0 {
                parts.push(format!("{} deleted", deleted_c));
            }
            if renamed_c > 0 {
                parts.push(format!("{} renamed", renamed_c));
            }

            format!("{} file(s) changed: {}", entries.len(), parts.join(", "))
        };

        let directories = scan_directories(&self.isolated_path)
            .map_err(|error| StructuredError::new("diff_scan_error", error.to_string(), false))?;
        let mut diff = WorkspaceDiff::new(self.view_id.clone(), entries, summary);
        diff.created_directories = directories
            .difference(&self.baseline_directories)
            .cloned()
            .collect();
        diff.deleted_directories = self
            .baseline_directories
            .difference(&directories)
            .cloned()
            .collect();
        Ok(diff)
    }

    /// Apply all modifications from the isolated workspace back to the base workspace.
    pub fn apply_to_base(&self) -> Result<(), StructuredError> {
        if self.isolation_mode == IsolationMode::Direct {
            return Ok(());
        }
        let diff = self.compute_diff()?;
        let current = scan_workspace_files(&self.base_path)
            .map_err(|error| StructuredError::new("merge_scan_error", error.to_string(), false))?;
        for entry in &diff.entries {
            let mut paths = vec![entry.path.as_path()];
            if let DiffKind::Renamed { from } = &entry.kind {
                paths.push(from);
            }
            for path in paths {
                if current.get(path) != self.baseline_files.get(path) {
                    return Err(StructuredError::new(
                        "workspace_conflict",
                        format!(
                            "Parent changed {} after sandbox allocation; isolated result retained",
                            path.display()
                        ),
                        true,
                    ));
                }
                let target = self.base_path.join(path);
                let mut ancestor = target.as_path();
                while !ancestor.exists() {
                    ancestor = ancestor.parent().ok_or_else(|| {
                        StructuredError::new("merge_path", "Invalid merge path", false)
                    })?;
                }
                if !ancestor
                    .canonicalize()
                    .map_err(|error| StructuredError::new("merge_path", error.to_string(), false))?
                    .starts_with(&self.base_path)
                {
                    return Err(StructuredError::new(
                        "merge_path_escape",
                        "Merge target leaves approved workspace",
                        false,
                    ));
                }
            }
        }
        for directory in &diff.created_directories {
            let target = self.base_path.join(directory);
            let mut ancestor = target.as_path();
            while !ancestor.exists() {
                ancestor = ancestor.parent().ok_or_else(|| {
                    StructuredError::new("merge_path", "Invalid directory", false)
                })?;
            }
            if !ancestor
                .canonicalize()
                .map_err(|error| StructuredError::new("merge_path", error.to_string(), false))?
                .starts_with(&self.base_path)
            {
                return Err(StructuredError::new(
                    "merge_path_escape",
                    "Directory leaves approved workspace",
                    false,
                ));
            }
            fs::create_dir_all(target).map_err(|error| {
                StructuredError::new("merge_directory", error.to_string(), false)
            })?;
        }

        for entry in &diff.entries {
            let base_file = self.base_path.join(&entry.path);
            let iso_file = self.isolated_path.join(&entry.path);

            match &entry.kind {
                DiffKind::Created | DiffKind::Modified => {
                    if let Some(parent) = base_file.parent() {
                        fs::create_dir_all(parent).map_err(|e| {
                            StructuredError::new(
                                "apply_error",
                                format!(
                                    "Failed to create parent directory {}: {}",
                                    parent.display(),
                                    e
                                ),
                                false,
                            )
                        })?;
                    }
                    fs::copy(&iso_file, &base_file).map_err(|e| {
                        StructuredError::new(
                            "apply_error",
                            format!("Failed to copy {} to base: {}", entry.path.display(), e),
                            false,
                        )
                    })?;
                }
                DiffKind::Deleted => {
                    if base_file.exists() {
                        fs::remove_file(&base_file).map_err(|e| {
                            StructuredError::new(
                                "apply_error",
                                format!("Failed to delete {}: {}", base_file.display(), e),
                                false,
                            )
                        })?;
                    }
                }
                DiffKind::Renamed { from } => {
                    let old_base_file = self.base_path.join(from);
                    if old_base_file.exists() {
                        fs::remove_file(&old_base_file).map_err(|e| {
                            StructuredError::new(
                                "apply_error",
                                format!(
                                    "Failed to remove old file {}: {}",
                                    old_base_file.display(),
                                    e
                                ),
                                false,
                            )
                        })?;
                    }
                    if let Some(parent) = base_file.parent() {
                        fs::create_dir_all(parent).map_err(|e| {
                            StructuredError::new(
                                "apply_error",
                                format!("Failed to create directory {}: {}", parent.display(), e),
                                false,
                            )
                        })?;
                    }
                    fs::copy(&iso_file, &base_file).map_err(|e| {
                        StructuredError::new(
                            "apply_error",
                            format!(
                                "Failed to copy renamed file {}: {}",
                                entry.path.display(),
                                e
                            ),
                            false,
                        )
                    })?;
                }
            }
        }
        for directory in diff.deleted_directories.iter().rev() {
            let target = self.base_path.join(directory);
            if target.exists()
                && target
                    .canonicalize()
                    .map_err(|error| StructuredError::new("merge_path", error.to_string(), false))?
                    .starts_with(&self.base_path)
            {
                fs::remove_dir(target).map_err(|error| {
                    StructuredError::new(
                        "workspace_conflict",
                        format!("Directory removal failed: {error}"),
                        true,
                    )
                })?;
            }
        }

        Ok(())
    }

    /// Clean up and remove the isolated workspace directory from disk.
    /// Direct views own the caller's workspace and must never remove it.
    /// Does not unlink the lease file to avoid inode races.
    pub fn clean(&self) -> Result<(), StructuredError> {
        if self.isolation_mode == IsolationMode::Direct {
            return Ok(());
        }
        if self.isolated_path.exists() {
            fs::remove_dir_all(&self.isolated_path).map_err(|e| {
                StructuredError::new(
                    "clean_error",
                    format!(
                        "Failed to remove isolated path {}: {}",
                        self.isolated_path.display(),
                        e
                    ),
                    false,
                )
            })?;
        }
        Ok(())
    }
}

/// Modification kind for a file within a workspace diff.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DiffKind {
    Created,
    Modified,
    Deleted,
    Renamed { from: PathBuf },
}

/// Entry representing a changed file in an isolated workspace with complete hash and byte metrics.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FileDiffEntry {
    pub path: PathBuf,
    pub kind: DiffKind,
    pub bytes_changed: usize,
    pub before_bytes: Option<u64>,
    pub after_bytes: Option<u64>,
    pub before_hash: Option<String>,
    pub after_hash: Option<String>,
}

/// Structured diff produced by an isolated workspace view on job or subagent completion.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceDiff {
    pub view_id: WorkspaceViewId,
    pub entries: Vec<FileDiffEntry>,
    pub summary: String,
    #[serde(default)]
    pub created_directories: Vec<PathBuf>,
    #[serde(default)]
    pub deleted_directories: Vec<PathBuf>,
}

impl WorkspaceDiff {
    pub fn empty(view_id: WorkspaceViewId) -> Self {
        Self {
            view_id,
            entries: Vec::new(),
            summary: "no changes".to_string(),
            created_directories: Vec::new(),
            deleted_directories: Vec::new(),
        }
    }

    pub fn new(
        view_id: WorkspaceViewId,
        entries: Vec<FileDiffEntry>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            view_id,
            entries,
            summary: summary.into(),
            created_directories: Vec::new(),
            deleted_directories: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
            && self.created_directories.is_empty()
            && self.deleted_directories.is_empty()
    }

    pub fn created_paths(&self) -> Vec<&Path> {
        self.entries
            .iter()
            .filter(|e| e.kind == DiffKind::Created)
            .map(|e| e.path.as_path())
            .collect()
    }

    pub fn deleted_paths(&self) -> Vec<&Path> {
        self.entries
            .iter()
            .filter(|e| e.kind == DiffKind::Deleted)
            .map(|e| e.path.as_path())
            .collect()
    }

    pub fn modified_paths(&self) -> Vec<&Path> {
        self.entries
            .iter()
            .filter(|e| e.kind == DiffKind::Modified)
            .map(|e| e.path.as_path())
            .collect()
    }

    pub fn renamed_paths(&self) -> Vec<(&Path, &Path)> {
        self.entries
            .iter()
            .filter_map(|e| match &e.kind {
                DiffKind::Renamed { from } => Some((from.as_path(), e.path.as_path())),
                _ => None,
            })
            .collect()
    }
}

/// Heavy dependency dirs are junctioned into the view instead of copied for
/// spawn speed. Writes through the junction land in the host workspace
/// immediately and are invisible to `compute_diff`, so junctioning is
/// opt-in via `OMP_ALLOW_HEAVY_JUNCTION_WRITE=1`; otherwise heavy dirs are
/// created empty in the view (jobs reinstall what they need).
fn heavy_junction_writes_allowed() -> bool {
    std::env::var("OMP_ALLOW_HEAVY_JUNCTION_WRITE")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn is_heavy_dir(name: &str) -> bool {
    matches!(
        name,
        "node_modules"
            | "target"
            | ".next"
            | ".turbo"
            | "dist"
            | "build"
            | "__pycache__"
            | ".venv"
            | ".cache"
            | "vendor"
    )
}
fn scan_directories(root: &Path) -> io::Result<std::collections::BTreeSet<PathBuf>> {
    fn visit(
        root: &Path,
        relative: &Path,
        result: &mut std::collections::BTreeSet<PathBuf>,
        depth: usize,
    ) -> io::Result<()> {
        if depth > 64 || result.len() > 100_000 {
            return Err(io::Error::other("Workspace directory budget exceeded"));
        }
        for entry in fs::read_dir(root.join(relative))? {
            let entry = entry?;
            let name_str = entry.file_name();
            let name_str = name_str.to_string_lossy();
            if matches!(name_str.as_ref(), ".git" | ".omp" | ".omp2")
                || is_heavy_dir(name_str.as_ref())
                || name_str.ends_with(".journal")
            {
                continue;
            }
            let metadata = fs::symlink_metadata(entry.path())?;
            if metadata.file_type().is_symlink() {
                continue;
            }
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                if metadata.file_attributes() & 0x400 != 0 {
                    continue;
                }
            }
            if metadata.is_dir() {
                let child = relative.join(entry.file_name());
                result.insert(child.clone());
                visit(root, &child, result, depth + 1)?;
            }
        }
        Ok(())
    }
    let mut result = Default::default();
    visit(root, Path::new(""), &mut result, 0)?;
    Ok(result)
}

/// Recursively scan workspace directory collecting relative paths, sizes, and SHA256 hashes.
/// Skips `.git` and detects directory cycles to prevent infinite loops.
fn scan_workspace_files(root: &Path) -> io::Result<HashMap<PathBuf, FileBaseline>> {
    let mut files = HashMap::new();
    if !root.exists() {
        return Ok(files);
    }
    let mut visited_dirs = HashSet::new();
    if let Ok(canon_root) = root.canonicalize() {
        visited_dirs.insert(canon_root);
    }
    scan_dir_recursive(root, Path::new(""), &mut visited_dirs, &mut files)?;
    Ok(files)
}

fn scan_dir_recursive(
    root: &Path,
    rel: &Path,
    visited_dirs: &mut HashSet<PathBuf>,
    results: &mut HashMap<PathBuf, FileBaseline>,
) -> io::Result<()> {
    let current = root.join(rel);
    for entry in fs::read_dir(&current)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();

        if matches!(name_str.as_ref(), ".git" | ".omp" | ".omp2")
            || is_heavy_dir(name_str.as_ref())
            || name_str.ends_with(".journal")
        {
            continue;
        }

        let child_rel = rel.join(&file_name);
        let symlink_meta = fs::symlink_metadata(entry.path())?;
        let file_type = symlink_meta.file_type();
        if file_type.is_symlink() {
            continue;
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            if symlink_meta.file_attributes() & 0x400 != 0 {
                continue;
            }
        }

        if file_type.is_dir() {
            if let Ok(canon_child) = entry.path().canonicalize()
                && !visited_dirs.insert(canon_child) {
                    // Cycle detected: skip to prevent infinite recursion
                    continue;
                }
            scan_dir_recursive(root, &child_rel, visited_dirs, results)?;
        } else if file_type.is_file() {
            let size = symlink_meta.len();
            let hash = compute_file_sha256(&entry.path())?;
            results.insert(child_rel, FileBaseline { size, sha256: hash });
        }
    }
    Ok(())
}

fn compute_file_sha256(path: &Path) -> io::Result<String> {
    let file = File::open(path)?;
    let mut reader = io::BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Recursively copy a directory into destination while preventing symlink/junction
/// cycles and traversal escaping the base directory.
fn copy_directory_safe(src: &Path, dst: &Path, _base_canon: &Path) -> io::Result<()> {
    let mut visited_dirs = HashSet::new();
    if let Ok(canon_src) = src.canonicalize() {
        visited_dirs.insert(canon_src);
    }
    copy_dir_recursive(src, dst, &mut visited_dirs)
}

fn copy_dir_recursive(src: &Path, dst: &Path, visited_dirs: &mut HashSet<PathBuf>) -> io::Result<()> {
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();

        if matches!(name_str.as_ref(), ".git" | ".omp" | ".omp2") || name_str.ends_with(".journal")
        {
            continue;
        }

        let entry_path = entry.path();
        let symlink_meta = fs::symlink_metadata(&entry_path)?;
        let file_type = symlink_meta.file_type();

        if file_type.is_symlink() {
            continue;
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            if symlink_meta.file_attributes() & 0x400 != 0 {
                continue;
            }
        }

        let target_path = dst.join(&file_name);

        if file_type.is_dir() {
            if is_heavy_dir(name_str.as_ref()) {
                if heavy_junction_writes_allowed() {
                    // Explicit opt-in: junction the host dependency tree into
                    // the view. Writes through the junction land in the host
                    // workspace immediately, and compute_diff/apply_to_base
                    // never see them because the scanners skip heavy dirs.
                    #[cfg(windows)]
                    {
                        let _ = std::process::Command::new("cmd")
                            .args([
                                "/c",
                                "mklink",
                                "/J",
                                target_path.to_str().unwrap_or(""),
                                entry_path.to_str().unwrap_or(""),
                            ])
                            .output();
                    }
                    #[cfg(not(windows))]
                    {
                        let _ = std::os::unix::fs::symlink(&entry_path, &target_path);
                    }
                } else {
                    // Default: present an empty dir so sandboxed jobs cannot
                    // mutate the host's dependency tree outside the lease.
                    let _ = fs::create_dir_all(&target_path);
                }
                continue;
            }
            if let Ok(canon_entry) = entry_path.canonicalize()
                && !visited_dirs.insert(canon_entry) {
                    // Cycle detected: skip
                    continue;
                }
            copy_dir_recursive(&entry_path, &target_path, visited_dirs)?;
        } else if file_type.is_file() {
            fs::copy(&entry_path, &target_path)?;
        }
    }

    Ok(())
}

/// Copy untracked files from base to isolated worktree view.
fn copy_untracked_files(base: &Path, isolated: &Path) -> io::Result<()> {
    // Same hardening as `allocate_internal`: never let repo config execute
    // hooks or LFS filters during a host-side copy.
    let output = Command::new("git")
        .args([
            "-c",
            "core.hooksPath=NUL",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.clean=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "ls-files",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(base)
        .output()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let src_file = base.join(trimmed);
            let dst_file = isolated.join(trimmed);
            if src_file.is_file() {
                if let Some(parent) = dst_file.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(&src_file, &dst_file)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_workspace_view_allocation_and_immutable_baseline_diff() {
        let root = std::env::temp_dir().join(format!(
            "omp_ws_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let base = root.join("base");
        let views = root.join("views");

        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("main.rs"), b"fn main() {}").unwrap();
        fs::write(base.join("README.txt"), b"documentation").unwrap();

        let session_id = SessionId::mint();
        let view = WorkspaceView::allocate(&base, &views, session_id.clone()).unwrap();

        // View directory exists and contains files
        assert!(view.isolated_path.exists());
        assert!(view.isolated_path.join("main.rs").exists());
        assert!(view.isolated_path.join("README.txt").exists());

        // Acquire OS exclusive lease
        let lease = view.acquire().unwrap();
        assert!(lease.lease_path().exists());

        // Concurrent acquire fails because OS lock is held
        let second_acquire = view.acquire();
        assert!(second_acquire.is_err());

        // Cannot allocate alias of parent
        let alias_err = WorkspaceView::allocate(&base, &base, session_id.clone());
        assert!(alias_err.is_err());

        // Cannot allocate views inside parent
        let inside_err = WorkspaceView::allocate(&base, base.join("views_sub"), session_id);
        assert!(inside_err.is_err());

        // Modify file in isolated workspace
        fs::write(
            view.isolated_path.join("main.rs"),
            b"fn main() { println!(); }",
        )
        .unwrap();
        // Create new file
        fs::write(view.isolated_path.join("added.rs"), b"pub fn helper() {}").unwrap();
        // Delete file
        fs::remove_file(view.isolated_path.join("README.txt")).unwrap();

        // Concurrently mutate the parent workspace to prove diff is against immutable baseline!
        fs::write(base.join("main.rs"), b"MUTATED PARENT WHILE CHILD WORKED").unwrap();
        fs::write(base.join("parent_only.rs"), b"new file in parent").unwrap();

        // Compute structured diff: must compare against baseline, NOT mutated parent!
        let diff = view.compute_diff().unwrap();
        assert_eq!(diff.entries.len(), 3);
        assert!(!diff.is_empty());

        let modified = diff.modified_paths();
        assert_eq!(modified.len(), 1);
        assert_eq!(modified[0], Path::new("main.rs"));

        let created = diff.created_paths();
        assert_eq!(created.len(), 1);
        assert_eq!(created[0], Path::new("added.rs"));

        let deleted = diff.deleted_paths();
        assert_eq!(deleted.len(), 1);
        assert_eq!(deleted[0], Path::new("README.txt"));

        // Release lease: closes file handle
        lease.release();

        // After release, acquiring lease succeeds again on same file (no stale marker deadlock!)
        let reacquired = view.acquire().unwrap();
        reacquired.release();

        // Clean up
        view.clean().unwrap();
        let _ = fs::remove_dir_all(&root);
    }
    #[test]
    fn direct_view_clean_never_removes_workspace() {
        let root = std::env::temp_dir().join(format!(
            "omp_ws_direct_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let base = root.join("base");
        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("sentinel.txt"), b"preserve").unwrap();

        let view = WorkspaceView::direct(&base, SessionId::mint()).unwrap();
        assert_eq!(view.base_path, view.isolated_path);
        assert_eq!(view.isolation_mode, IsolationMode::Direct);
        view.clean().unwrap();

        assert_eq!(
            fs::read_to_string(base.join("sentinel.txt")).unwrap(),
            "preserve"
        );
        let _ = fs::remove_dir_all(&root);
    }
}
