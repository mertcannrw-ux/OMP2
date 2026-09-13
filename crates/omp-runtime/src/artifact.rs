use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use omp_types::{ActorId, ArtifactId, JobId, SessionId, StructuredError, ToolCallId};

/// Validate that a content hash string is strictly 64 lowercase hexadecimal characters
/// to prevent path traversal or malformed blob lookup.
fn validate_content_hash(hash: &str) -> Result<(), ArtifactError> {
    if hash.len() != 64 || !hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(ArtifactError::Invalid(format!(
            "Invalid content hash {:?}: must be exactly 64 lowercase hexadecimal characters",
            hash
        )));
    }
    Ok(())
}

/// Stream a file on disk and return its lowercase SHA-256 hex digest.
fn hash_file(path: &Path) -> Result<String, ArtifactError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
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

/// Origin that produced the artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactOrigin {
    Job(JobId),
    ToolCall(ToolCallId),
    Subagent(ActorId),
    User,
    System,
}

/// Retention policy governing artifact lifetime.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactRetention {
    /// Discarded when job completes.
    Transient,
    /// Preserved for the duration of the parent session.
    Session,
    /// Saved persistently in storage.
    Persistent,
    /// Pinned by user or policy; will not be pruned automatically.
    Pinned,
}

/// Access control scope for the artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactScope {
    Session,
    Job(JobId),
    Actor(ActorId),
    Global,
}

/// Scope credential presented by a caller to access or store artifacts.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScopeCredential {
    pub session_id: Option<SessionId>,
    pub actor_id: Option<ActorId>,
    pub job_id: Option<JobId>,
    pub is_host: bool,
}

impl ScopeCredential {
    pub fn host() -> Self {
        Self {
            session_id: None,
            actor_id: None,
            job_id: None,
            is_host: true,
        }
    }

    /// Unprivileged credential with access only to Global-scoped artifacts.
    /// Never grants host privilege escalation.
    pub fn global() -> Self {
        Self {
            session_id: None,
            actor_id: None,
            job_id: None,
            is_host: false,
        }
    }

    pub fn session(session_id: SessionId) -> Self {
        Self {
            session_id: Some(session_id),
            actor_id: None,
            job_id: None,
            is_host: false,
        }
    }

    pub fn actor(session_id: Option<SessionId>, actor_id: ActorId) -> Self {
        Self {
            session_id,
            actor_id: Some(actor_id),
            job_id: None,
            is_host: false,
        }
    }

    pub fn job(session_id: Option<SessionId>, job_id: JobId) -> Self {
        Self {
            session_id,
            actor_id: None,
            job_id: Some(job_id),
            is_host: false,
        }
    }

    /// Check if this credential authorizes access to an artifact with the given journal-derived metadata.
    pub fn allows(&self, metadata: &ArtifactMetadata) -> bool {
        if self.is_host {
            return true;
        }
        if metadata.scope == ArtifactScope::Global {
            return true;
        }

        // Cross-session check: if both state a session_id and they do not match, deny.
        if let (Some(caller_sid), Some(artifact_sid)) = (&self.session_id, &metadata.session_id)
            && caller_sid != artifact_sid {
                return false;
            }

        match &metadata.scope {
            ArtifactScope::Global => true,
            ArtifactScope::Session => {
                // Must have explicit session_id on both caller and metadata; missing session_id MUST deny!
                if let (Some(caller_sid), Some(artifact_sid)) =
                    (&self.session_id, &metadata.session_id)
                {
                    caller_sid == artifact_sid
                } else {
                    false
                }
            }
            ArtifactScope::Actor(expected_actor) => self.actor_id.as_ref() == Some(expected_actor),
            ArtifactScope::Job(expected_job) => self.job_id.as_ref() == Some(expected_job),
        }
    }

    /// Check if this credential allows storing an artifact with the given scope and session.
    /// Only host may store with ArtifactScope::Global.
    pub fn allows_store(&self, scope: &ArtifactScope, session_id: Option<&SessionId>) -> bool {
        if self.is_host {
            return true;
        }
        match scope {
            // Only host may publish globally!
            ArtifactScope::Global => false,
            ArtifactScope::Session => {
                if let (Some(caller_sid), Some(target_sid)) = (&self.session_id, session_id) {
                    caller_sid == target_sid
                } else {
                    false
                }
            }
            ArtifactScope::Actor(actor) => self.actor_id.as_ref() == Some(actor),
            ArtifactScope::Job(job) => self.job_id.as_ref() == Some(job),
        }
    }
}

/// Errors occurring during artifact store operations.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("Artifact access denied: {0}")]
    AccessDenied(String),
    #[error("Artifact not found: {0}")]
    NotFound(String),
    #[error("Artifact limit exceeded: {0}")]
    LimitExceeded(String),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Corrupted artifact hash: expected {expected}, computed {computed}")]
    HashMismatch { expected: String, computed: String },
    #[error("Corrupted artifact data: {0}")]
    Corrupted(String),
    #[error("Invalid artifact operation: {0}")]
    Invalid(String),
}

impl From<ArtifactError> for StructuredError {
    fn from(err: ArtifactError) -> Self {
        match &err {
            ArtifactError::AccessDenied(msg) => {
                StructuredError::new("artifact_access_denied", msg, false)
            }
            ArtifactError::NotFound(msg) => StructuredError::new("artifact_not_found", msg, false),
            ArtifactError::LimitExceeded(msg) => {
                StructuredError::new("artifact_limit_exceeded", msg, false)
            }
            ArtifactError::HashMismatch { expected, computed } => StructuredError::new(
                "artifact_hash_mismatch",
                format!("expected hash {expected}, got {computed}"),
                false,
            ),
            ArtifactError::Corrupted(msg) => StructuredError::new("artifact_corrupted", msg, false),
            ArtifactError::Io(err) => {
                StructuredError::new("artifact_io_error", err.to_string(), false)
            }
            ArtifactError::Invalid(msg) => StructuredError::new("artifact_invalid", msg, false),
        }
    }
}

/// Complete metadata for an artifact.
/// Persisted authoritative state stays strictly in the caller's session DOM / journal.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactMetadata {
    pub id: ArtifactId,
    pub content_hash: String,
    pub media_type: String,
    pub byte_length: usize,
    pub origin: ArtifactOrigin,
    pub retention: ArtifactRetention,
    pub scope: ArtifactScope,
    pub created_at_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<SessionId>,
}

impl ArtifactMetadata {
    /// Return the model-facing canonical reference URI (`artifact://<id>`).
    pub fn uri(&self) -> String {
        format!("artifact://{}", self.id.as_str())
    }
}

/// Disk-backed materialized artifact handle pointing to content on disk.
/// Never loads full payload into memory by default.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Artifact {
    pub metadata: ArtifactMetadata,
    pub blob_path: PathBuf,
}

impl Artifact {
    pub fn new(metadata: ArtifactMetadata, blob_path: PathBuf) -> Self {
        Self {
            metadata,
            blob_path,
        }
    }

    pub fn id(&self) -> &ArtifactId {
        &self.metadata.id
    }

    pub fn uri(&self) -> String {
        self.metadata.uri()
    }

    /// Open a streaming file handle to the underlying disk blob.
    pub fn open(&self) -> io::Result<File> {
        File::open(&self.blob_path)
    }

    /// Read at most `max_bytes` from the disk blob into memory.
    pub fn read_bounded(&self, max_bytes: usize) -> Result<Vec<u8>, ArtifactError> {
        let file = self.open()?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        let mut bounded = io::Read::take(&mut reader, max_bytes as u64);
        bounded.read_to_end(&mut buffer)?;
        Ok(buffer)
    }
}

/// Host-owned disk-backed artifact content store with content-addressed storage.
///
/// Authority lives entirely in caller-provided journal-derived `ArtifactMetadata`.
/// The store only houses immutable content-addressed blobs keyed by SHA256 content hashes.
#[derive(Debug)]
pub struct ArtifactStore {
    root: PathBuf,
    blobs_dir: PathBuf,
    tmp_dir: PathBuf,
}

impl ArtifactStore {
    /// Open or initialize an artifact store rooted at the given path.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, ArtifactError> {
        let root = root.as_ref().to_path_buf();
        let blobs_dir = root.join("blobs");
        let tmp_dir = root.join("tmp");

        fs::create_dir_all(&blobs_dir)?;
        fs::create_dir_all(&tmp_dir)?;

        Ok(Self {
            root,
            blobs_dir,
            tmp_dir,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn blobs_dir(&self) -> &Path {
        &self.blobs_dir
    }

    /// Content-addressed blob path for a given validated SHA256 hex digest.
    pub fn blob_path(&self, content_hash: &str) -> Result<PathBuf, ArtifactError> {
        validate_content_hash(content_hash)?;
        Ok(self.blobs_dir.join(format!("{}.blob", content_hash)))
    }

    /// Check if a blob with the given content hash exists in the content store.
    pub fn has_blob(&self, content_hash: &str) -> bool {
        match self.blob_path(content_hash) {
            Ok(p) => p.exists(),
            Err(_) => false,
        }
    }

    /// Store an artifact from an input stream (`io::Read`), writing chunks directly
    /// to disk while calculating a running SHA256 hash. Enforces `max_bytes` without
    /// ever collecting the full artifact Vec in RAM.
    ///
    /// Returns the complete `ArtifactMetadata` for caller to record in the session DOM/journal.
    #[allow(clippy::too_many_arguments)]
    pub fn store_stream<R: Read>(
        &self,
        id: ArtifactId,
        reader: &mut R,
        max_bytes: usize,
        media_type: impl Into<String>,
        origin: ArtifactOrigin,
        retention: ArtifactRetention,
        scope: ArtifactScope,
        credential: &ScopeCredential,
    ) -> Result<ArtifactMetadata, ArtifactError> {
        if !credential.allows_store(&scope, credential.session_id.as_ref()) {
            return Err(ArtifactError::AccessDenied(format!(
                "Credential does not have permission to store artifact with scope {:?}",
                scope
            )));
        }

        let now_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let tmp_filename = format!("spool_{}_{}.tmp", id.as_str(), now_nanos);
        let tmp_path = self.tmp_dir.join(tmp_filename);

        let mut hasher = Sha256::new();
        let mut total_bytes = 0usize;

        {
            let tmp_file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(tmp_file);
            let mut chunk = [0u8; 64 * 1024];

            loop {
                let bytes_read = match reader.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => {
                        let _ = fs::remove_file(&tmp_path);
                        return Err(ArtifactError::Io(e));
                    }
                };

                if total_bytes.saturating_add(bytes_read) > max_bytes {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(ArtifactError::LimitExceeded(format!(
                        "Artifact size exceeded limit of {} bytes (read {} bytes so far)",
                        max_bytes,
                        total_bytes.saturating_add(bytes_read)
                    )));
                }

                if let Err(e) = writer.write_all(&chunk[..bytes_read]) {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(ArtifactError::Io(e));
                }

                hasher.update(&chunk[..bytes_read]);
                total_bytes += bytes_read;
            }

            if let Err(e) = writer.flush() {
                let _ = fs::remove_file(&tmp_path);
                return Err(ArtifactError::Io(e));
            }
        }

        let content_hash = hex::encode(hasher.finalize());
        let target_blob_path = self.blob_path(&content_hash)?;

        // Content-addressed deduplication: if blob already exists, drop the temp file
        if target_blob_path.exists() {
            let _ = fs::remove_file(&tmp_path);
        } else if fs::rename(&tmp_path, &target_blob_path).is_err() {
            // Fallback for cross-device renames: copy then remove
            fs::copy(&tmp_path, &target_blob_path)?;
            let _ = fs::remove_file(&tmp_path);
        }

        let created_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let metadata = ArtifactMetadata {
            id,
            content_hash,
            media_type: media_type.into(),
            byte_length: total_bytes,
            origin,
            retention,
            scope,
            created_at_ms,
            session_id: credential.session_id.clone(),
        };

        Ok(metadata)
    }

    /// Store an artifact from in-memory byte slice, enforcing `max_bytes`
    /// like the streaming path (previously this bypassed quotas by passing
    /// `data.len() + 1` as the limit).
    #[allow(clippy::too_many_arguments)]
    pub fn store_bytes(
        &self,
        id: ArtifactId,
        data: &[u8],
        max_bytes: usize,
        media_type: impl Into<String>,
        origin: ArtifactOrigin,
        retention: ArtifactRetention,
        scope: ArtifactScope,
        credential: &ScopeCredential,
    ) -> Result<ArtifactMetadata, ArtifactError> {
        if data.len() > max_bytes {
            return Err(ArtifactError::LimitExceeded(format!(
                "in-memory artifact ({} bytes) exceeds limit of {max_bytes} bytes",
                data.len()
            )));
        }
        let mut slice_reader = data;
        self.store_stream(
            id,
            &mut slice_reader,
            max_bytes,
            media_type,
            origin,
            retention,
            scope,
            credential,
        )
    }

    /// Ingest an already existing spooled file directly into the content-addressed store
    /// by moving/linking it into place.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_spooled_file(
        &self,
        id: ArtifactId,
        source_path: &Path,
        expected_hash: &str,
        byte_length: usize,
        media_type: impl Into<String>,
        origin: ArtifactOrigin,
        retention: ArtifactRetention,
        scope: ArtifactScope,
        credential: &ScopeCredential,
    ) -> Result<ArtifactMetadata, ArtifactError> {
        if !credential.allows_store(&scope, credential.session_id.as_ref()) {
            return Err(ArtifactError::AccessDenied(format!(
                "Credential does not have permission to store artifact with scope {:?}",
                scope
            )));
        }

        let target_blob_path = self.blob_path(expected_hash)?;
        if !target_blob_path.exists() {
            if fs::rename(source_path, &target_blob_path).is_err() {
                fs::copy(source_path, &target_blob_path)?;
                let _ = fs::remove_file(source_path);
            }
            // The caller-supplied hash is not trusted: verify the placed blob
            // byte-for-byte before it becomes addressable content.
            let actual = hash_file(&target_blob_path)?;
            if actual != expected_hash {
                let _ = fs::remove_file(&target_blob_path);
                return Err(ArtifactError::Corrupted(format!(
                    "Spooled file hash mismatch: expected {expected_hash}, got {actual}"
                )));
            }
        } else {
            let _ = fs::remove_file(source_path);
        }

        let created_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let metadata = ArtifactMetadata {
            id,
            content_hash: expected_hash.to_string(),
            media_type: media_type.into(),
            byte_length,
            origin,
            retention,
            scope,
            created_at_ms,
            session_id: credential.session_id.clone(),
        };

        Ok(metadata)
    }

    /// Retrieve an Artifact handle for reading given journal-derived metadata after credential check.
    /// Validates hash format and ensures disk file length matches metadata.
    pub fn get(
        &self,
        metadata: &ArtifactMetadata,
        credential: &ScopeCredential,
    ) -> Result<Artifact, ArtifactError> {
        if !credential.allows(metadata) {
            return Err(ArtifactError::AccessDenied(format!(
                "Access to artifact {} denied for provided credential",
                metadata.id.as_str()
            )));
        }

        let blob_path = self.blob_path(&metadata.content_hash)?;
        if !blob_path.exists() {
            return Err(ArtifactError::NotFound(format!(
                "Underlying blob for artifact {} (hash {}) missing on disk",
                metadata.id.as_str(),
                metadata.content_hash
            )));
        }

        // Verify disk file size matches metadata byte_length
        let disk_len = fs::metadata(&blob_path).map_err(ArtifactError::Io)?.len() as usize;
        if disk_len != metadata.byte_length {
            return Err(ArtifactError::Corrupted(format!(
                "Artifact size mismatch for {}: metadata states {} bytes, disk file is {} bytes",
                metadata.id.as_str(),
                metadata.byte_length,
                disk_len
            )));
        }

        Ok(Artifact::new(metadata.clone(), blob_path))
    }

    /// Open a bounded stream file reader for an artifact using journal-derived metadata after credential check.
    pub fn open_read(
        &self,
        metadata: &ArtifactMetadata,
        credential: &ScopeCredential,
    ) -> Result<File, ArtifactError> {
        let artifact = self.get(metadata, credential)?;
        artifact.open().map_err(ArtifactError::Io)
    }

    /// Read at most `max_bytes` of an artifact using journal-derived metadata after credential check.
    /// Verifies SHA256 integrity when the full payload is read.
    pub fn read_bounded(
        &self,
        metadata: &ArtifactMetadata,
        max_bytes: usize,
        credential: &ScopeCredential,
    ) -> Result<Vec<u8>, ArtifactError> {
        let artifact = self.get(metadata, credential)?;
        let data = artifact.read_bounded(max_bytes)?;

        if data.len() == metadata.byte_length {
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let computed = hex::encode(hasher.finalize());
            if computed != metadata.content_hash {
                return Err(ArtifactError::HashMismatch {
                    expected: metadata.content_hash.clone(),
                    computed,
                });
            }
        }

        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omp_types::LimitPolicy;

    #[test]
    fn test_disk_backed_hashed_storage_and_read() {
        let temp_dir = std::env::temp_dir().join(format!(
            "omp_art_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let store = ArtifactStore::open(&temp_dir).unwrap();

        let id = ArtifactId::mint();
        let payload = b"Hello, disk-backed hashed artifact storage world!";
        let cred = ScopeCredential::host();

        let meta = store
            .store_bytes(
                id.clone(),
                payload,
                LimitPolicy::default().max_artifact_bytes,
                "text/plain",
                ArtifactOrigin::System,
                ArtifactRetention::Transient,
                ArtifactScope::Global,
                &cred,
            )
            .unwrap();

        assert_eq!(meta.byte_length, payload.len());
        assert_eq!(meta.content_hash.len(), 64);

        let blob_path = store.blob_path(&meta.content_hash).unwrap();
        assert!(blob_path.exists());
        let read_back = fs::read(&blob_path).unwrap();
        assert_eq!(read_back, payload);

        // Read bounded using journal metadata
        let bounded = store.read_bounded(&meta, 1024, &cred).unwrap();
        assert_eq!(bounded, payload);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_hostile_content_hash_path_traversal_rejected() {
        let temp_dir = std::env::temp_dir().join(format!(
            "omp_art_trav_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let store = ArtifactStore::open(&temp_dir).unwrap();

        // Hostile content hash attempting path traversal
        assert!(store.blob_path("../../etc/passwd").is_err());
        assert!(store.blob_path("..\\..\\Windows").is_err());
        assert!(store.blob_path("not_64_chars").is_err());
        assert!(
            store
                .blob_path("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
                .is_err()
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_global_scope_storage_restricted_to_host() {
        let temp_dir = std::env::temp_dir().join(format!(
            "omp_art_glob_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let store = ArtifactStore::open(&temp_dir).unwrap();

        let non_host_cred = ScopeCredential::session(SessionId::mint());

        // Non-host cannot publish Global artifacts
        let store_res = store.store_bytes(
            ArtifactId::mint(),
            b"global payload",
            LimitPolicy::default().max_artifact_bytes,
            "text/plain",
            ArtifactOrigin::User,
            ArtifactRetention::Persistent,
            ArtifactScope::Global,
            &non_host_cred,
        );
        assert!(matches!(store_res, Err(ArtifactError::AccessDenied(_))));

        // Host CAN publish Global artifacts
        let host_cred = ScopeCredential::host();
        let host_store = store.store_bytes(
            ArtifactId::mint(),
            b"global payload",
            LimitPolicy::default().max_artifact_bytes,
            "text/plain",
            ArtifactOrigin::User,
            ArtifactRetention::Persistent,
            ArtifactScope::Global,
            &host_cred,
        );
        assert!(host_store.is_ok());

        // Oversized in-memory payloads are rejected instead of bypassing quota.
        let oversized = store.store_bytes(
            ArtifactId::mint(),
            b"global payload",
            4,
            "text/plain",
            ArtifactOrigin::User,
            ArtifactRetention::Persistent,
            ArtifactScope::Global,
            &host_cred,
        );
        assert!(matches!(
            oversized,
            Err(ArtifactError::LimitExceeded(_))
        ));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_ingest_rejects_spool_with_wrong_declared_hash() {
        let temp_dir = std::env::temp_dir().join(format!(
            "omp_art_ingest_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let store = ArtifactStore::open(&temp_dir).unwrap();

        let spool = temp_dir.join("spooled.bin");
        fs::write(&spool, b"real artifact content").unwrap();
        let real_hash = hash_file(&spool).unwrap();

        let cred = ScopeCredential::host();

        // Wrong declared hash: ingest must fail and leave no blob behind.
        let bogus = "0".repeat(64);
        let err = store
            .ingest_spooled_file(
                ArtifactId::mint(),
                &spool,
                &bogus,
                fs::metadata(&spool).unwrap().len() as usize,
                "text/plain",
                ArtifactOrigin::System,
                ArtifactRetention::Session,
                ArtifactScope::Session,
                &cred,
            )
            .unwrap_err();
        assert!(matches!(err, ArtifactError::Corrupted(_)));
        assert!(!store.blob_path(&bogus).unwrap().exists());
        // Ingest consumes the source spool on every path; the failed blob
        // must not remain in the content store either.

        // Correct declared hash: ingest succeeds and content is readable.
        fs::write(&spool, b"real artifact content").unwrap();
        let meta = store
            .ingest_spooled_file(
                ArtifactId::mint(),
                &spool,
                &real_hash,
                fs::metadata(&spool).unwrap().len() as usize,
                "text/plain",
                ArtifactOrigin::System,
                ArtifactRetention::Session,
                ArtifactScope::Session,
                &cred,
            )
            .unwrap();
        assert_eq!(store.read_bounded(&meta, 1024, &cred).unwrap(), b"real artifact content");

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
