use hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use omp_types::{ArtifactId, LimitPolicy, TruncationDiag};

use crate::artifact::{
    ArtifactError, ArtifactMetadata, ArtifactOrigin, ArtifactRetention, ArtifactScope,
    ArtifactStore, ScopeCredential,
};

/// The kind of stream being accumulated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum StreamKind {
    Stdout,
    Stderr,
    Combined,
}

/// A single chunk received from a stream.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct StreamChunk {
    pub kind: StreamKind,
    pub data: Vec<u8>,
    pub timestamp_ms: u64,
}

/// Summary of a completed disk spool for artifact materialization.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SpoolSummary {
    pub path: PathBuf,
    pub content_hash: String,
    pub byte_length: usize,
    pub truncated: bool,
}

/// Disk-backed spool writer that streams bytes directly to a temporary file
/// computing a running SHA256 hash without loading the whole stream into memory.
#[derive(Debug)]
pub struct StreamSpool {
    writer: Option<BufWriter<File>>,
    path: PathBuf,
    hasher: Sha256,
    spooled_bytes: usize,
    max_artifact_bytes: usize,
    spool_truncated: bool,
}

impl StreamSpool {
    /// Create a new disk spool in the given directory (or system temp).
    pub fn new(spool_dir: Option<&Path>, max_artifact_bytes: usize) -> io::Result<Self> {
        let dir = match spool_dir {
            Some(d) => d.to_path_buf(),
            None => std::env::temp_dir().join("omp_spool"),
        };
        fs::create_dir_all(&dir)?;

        let path = dir.join(format!("spool-{}.tmp", omp_types::ArtifactId::mint()));
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            writer: Some(writer),
            path,
            hasher: Sha256::new(),
            spooled_bytes: 0,
            max_artifact_bytes,
            spool_truncated: false,
        })
    }

    /// Stream a chunk to the disk spool, updating hash and respecting max_artifact_bytes.
    pub fn write_chunk(&mut self, chunk: &[u8]) -> io::Result<()> {
        if self.spool_truncated {
            return Ok(());
        }

        let writer = match &mut self.writer {
            Some(w) => w,
            None => {
                return Err(io::Error::other(
                    "Spool writer already closed",
                ));
            }
        };

        let remaining = self.max_artifact_bytes.saturating_sub(self.spooled_bytes);
        if remaining == 0 {
            self.spool_truncated = true;
            return Ok(());
        }

        let to_write = chunk.len().min(remaining);
        if to_write < chunk.len() {
            self.spool_truncated = true;
        }

        if to_write > 0 {
            writer.write_all(&chunk[..to_write])?;
            self.hasher.update(&chunk[..to_write]);
            self.spooled_bytes += to_write;
        }

        Ok(())
    }

    pub fn spooled_bytes(&self) -> usize {
        self.spooled_bytes
    }

    pub fn is_spool_truncated(&self) -> bool {
        self.spool_truncated
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Finish writing, flush the buffer, and compute the final SHA256 digest.
    pub fn finish(mut self) -> io::Result<SpoolSummary> {
        if let Some(mut w) = self.writer.take() {
            w.flush()?;
            w.get_ref().sync_all()?;
        }

        let content_hash = hex::encode(std::mem::take(&mut self.hasher).finalize());
        Ok(SpoolSummary {
            path: std::mem::take(&mut self.path),
            content_hash,
            byte_length: self.spooled_bytes,
            truncated: self.spool_truncated,
        })
    }
}

impl Drop for StreamSpool {
    fn drop(&mut self) {
        // If writer is still open, spool was not finalized; clean up temp file
        if self.writer.take().is_some() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// A bounded stream buffer that accumulates output up to configured limits
/// in an in-memory resident buffer while concurrently spooling to disk with a running hash.
#[derive(Debug)]
pub struct BoundedStream {
    kind: StreamKind,
    max_bytes: usize,
    max_lines: usize,
    buffer: Vec<u8>,
    spool: Option<StreamSpool>,
    spool_error: Option<String>,
    observed_bytes: usize,
    observed_lines: usize,
    accepted_lines: usize,
    truncated: bool,
}

impl BoundedStream {
    pub fn new(kind: StreamKind, max_bytes: usize, max_lines: usize, capture_spill: bool) -> Self {
        Self::new_with_limits(
            kind,
            max_bytes,
            max_lines,
            100 * 1024 * 1024,
            capture_spill,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_limits(
        kind: StreamKind,
        max_bytes: usize,
        max_lines: usize,
        max_artifact_bytes: usize,
        capture_spill: bool,
        spool_dir: Option<PathBuf>,
    ) -> Self {
        let (spool, spool_error) = if capture_spill {
            match StreamSpool::new(spool_dir.as_deref(), max_artifact_bytes) {
                Ok(s) => (Some(s), None),
                Err(e) => (
                    None,
                    Some(format!("Failed to initialize disk spool: {}", e)),
                ),
            }
        } else {
            (None, None)
        };

        Self {
            kind,
            max_bytes,
            max_lines,
            buffer: Vec::new(),
            spool,
            spool_error,
            observed_bytes: 0,
            observed_lines: 0,
            accepted_lines: 0,
            truncated: false,
        }
    }

    /// Push a chunk of bytes into the stream. Updates observed metrics,
    /// writes to disk spool if active, and accumulates in memory up to limits.
    pub fn push(&mut self, chunk: &[u8]) {
        let chunk_lines = chunk.iter().filter(|&&b| b == b'\n').count();
        self.observed_bytes = self.observed_bytes.saturating_add(chunk.len());
        self.observed_lines = self.observed_lines.saturating_add(chunk_lines);

        // Spool to disk with running hash
        if let Some(spool) = &mut self.spool
            && let Err(e) = spool.write_chunk(chunk) {
                self.spool_error = Some(format!("Failed writing to disk spool: {}", e));
                self.spool = None;
            }

        // Bounded resident in-memory accumulation
        if self.truncated {
            return;
        }

        let current_bytes = self.buffer.len();
        let remaining_bytes = self.max_bytes.saturating_sub(current_bytes);

        if remaining_bytes == 0 && !chunk.is_empty() {
            self.truncated = true;
            return;
        }

        let bytes_to_take = chunk.len().min(remaining_bytes);
        let slice_to_add = &chunk[..bytes_to_take];

        // Check line limit within the accepted slice
        let mut accepted_bytes = 0;
        let mut lines_count = self.accepted_lines;

        for &b in slice_to_add {
            if b == b'\n' {
                lines_count += 1;
                if lines_count > self.max_lines {
                    self.truncated = true;
                    break;
                }
            }
            accepted_bytes += 1;
        }

        self.buffer
            .extend_from_slice(&slice_to_add[..accepted_bytes]);
        self.accepted_lines = lines_count.min(self.max_lines);

        if bytes_to_take < chunk.len() || accepted_bytes < bytes_to_take {
            self.truncated = true;
        }
    }

    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    pub fn observed_bytes(&self) -> usize {
        self.observed_bytes
    }

    pub fn observed_lines(&self) -> usize {
        self.observed_lines
    }

    /// Access the in-memory bounded resident buffer.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.buffer).to_string()
    }

    pub fn spool_path(&self) -> Option<&Path> {
        self.spool.as_ref().map(|s| s.path())
    }

    pub fn is_spool_truncated(&self) -> bool {
        self.spool
            .as_ref()
            .is_some_and(|s| s.is_spool_truncated())
    }

    /// Finalize the disk spool and return summary metrics and file path.
    pub fn finish_spool(&mut self) -> Option<SpoolSummary> {
        self.spool.take().and_then(|s| s.finish().ok())
    }

    /// Ingest the disk spool directly into an ArtifactStore without loading into memory.
    #[allow(clippy::too_many_arguments)]
    pub fn persist_to_artifact_store(
        &mut self,
        store: &ArtifactStore,
        id: ArtifactId,
        media_type: impl Into<String>,
        origin: ArtifactOrigin,
        retention: ArtifactRetention,
        scope: ArtifactScope,
        credential: &ScopeCredential,
    ) -> Result<Option<ArtifactMetadata>, ArtifactError> {
        if let Some(err) = &self.spool_error {
            return Err(ArtifactError::Io(io::Error::other(
                err.clone(),
            )));
        }
        let summary = match self.spool.take() {
            Some(s) => s.finish().map_err(ArtifactError::Io)?,
            None => return Ok(None),
        };

        let meta = store.ingest_spooled_file(
            id,
            &summary.path,
            &summary.content_hash,
            summary.byte_length,
            media_type,
            origin,
            retention,
            scope,
            credential,
        )?;

        Ok(Some(meta))
    }

    /// Produce a TruncationDiag if this stream was truncated.
    pub fn truncation_diag(
        &self,
        artifact_id: Option<ArtifactId>,
        fetchable: bool,
    ) -> Option<TruncationDiag> {
        if !self.truncated {
            return None;
        }
        Some(TruncationDiag::new(
            self.max_bytes,
            self.observed_bytes,
            self.max_lines,
            self.observed_lines,
            artifact_id,
            fetchable,
            format!("{:?} stream truncated after exceeding limits", self.kind),
        ))
    }
}

/// Context for persisting spooled stream output into an ArtifactStore.
#[derive(Clone, Debug)]
pub struct AccumulatorStoreContext {
    pub store: Arc<ArtifactStore>,
    pub origin: ArtifactOrigin,
    pub scope: ArtifactScope,
    pub credential: ScopeCredential,
}

/// Unified bounded stream accumulator containing both stdout and stderr buffers,
/// along with overall crossing event counting and disk spool persistence.
#[derive(Debug)]
pub struct BoundedStreamAccumulator {
    pub stdout: BoundedStream,
    pub stderr: BoundedStream,
    event_count: usize,
    max_events: usize,
    events_truncated: bool,
    store_context: Option<AccumulatorStoreContext>,
}

impl BoundedStreamAccumulator {
    pub fn new(limits: &LimitPolicy, capture_spill: bool) -> Self {
        Self::with_spool_dir(limits, capture_spill, None)
    }

    pub fn with_spool_dir(
        limits: &LimitPolicy,
        capture_spill: bool,
        spool_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            stdout: BoundedStream::new_with_limits(
                StreamKind::Stdout,
                limits.max_bytes,
                limits.max_lines,
                limits.max_artifact_bytes,
                capture_spill,
                spool_dir.clone(),
            ),
            stderr: BoundedStream::new_with_limits(
                StreamKind::Stderr,
                limits.max_bytes,
                limits.max_lines,
                limits.max_artifact_bytes,
                capture_spill,
                spool_dir,
            ),
            event_count: 0,
            max_events: limits.max_events,
            events_truncated: false,
            store_context: None,
        }
    }

    pub fn with_store(
        limits: &LimitPolicy,
        store: Arc<ArtifactStore>,
        origin: ArtifactOrigin,
        scope: ArtifactScope,
        credential: ScopeCredential,
    ) -> Self {
        let mut acc = Self::new(limits, true);
        acc.store_context = Some(AccumulatorStoreContext {
            store,
            origin,
            scope,
            credential,
        });
        acc
    }
    pub fn push_stdout(&mut self, chunk: &[u8]) {
        self.stdout.push(chunk);
    }

    pub fn push_stderr(&mut self, chunk: &[u8]) {
        self.stderr.push(chunk);
    }
    /// Push a chunk matching its StreamKind (Stdout or Stderr).
    pub fn push_chunk(&mut self, chunk: &StreamChunk) {
        match chunk.kind {
            StreamKind::Stdout => self.push_stdout(&chunk.data),
            StreamKind::Stderr => self.push_stderr(&chunk.data),
            StreamKind::Combined => {
                self.push_stdout(&chunk.data);
            }
        }
    }

    /// Persist both stdout and stderr spools (if spooled data exists) into the given store.
    pub fn persist_artifacts(
        &mut self,
        store: &ArtifactStore,
        origin: ArtifactOrigin,
        scope: ArtifactScope,
        credential: &ScopeCredential,
    ) -> Result<Vec<ArtifactMetadata>, ArtifactError> {
        let mut results = Vec::new();
        if self.stdout.spool_path().is_some() && self.stdout.observed_bytes() > 0 {
            let id = ArtifactId::mint();
            if let Some(meta) = self.persist_stdout_artifact(
                store,
                id,
                origin.clone(),
                ArtifactRetention::Session,
                scope.clone(),
                credential,
            )? {
                results.push(meta);
            }
        }
        if self.stderr.spool_path().is_some() && self.stderr.observed_bytes() > 0 {
            let id = ArtifactId::mint();
            if let Some(meta) = self.persist_stderr_artifact(
                store,
                id,
                origin,
                ArtifactRetention::Session,
                scope,
                credential,
            )? {
                results.push(meta);
            }
        }
        Ok(results)
    }

    /// Finalize and ingest spooled artifacts using the store context configured via `with_store`.
    pub fn finish_artifacts(&mut self) -> Result<Vec<ArtifactMetadata>, ArtifactError> {
        if let Some(ctx) = self.store_context.clone() {
            self.persist_artifacts(&ctx.store, ctx.origin, ctx.scope, &ctx.credential)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn record_event(&mut self) -> bool {
        self.event_count += 1;
        if self.event_count > self.max_events {
            self.events_truncated = true;
            false
        } else {
            true
        }
    }

    pub fn event_count(&self) -> usize {
        self.event_count
    }

    pub fn is_truncated(&self) -> bool {
        self.stdout.is_truncated() || self.stderr.is_truncated() || self.events_truncated
    }

    pub fn stdout_lossy(&self) -> String {
        self.stdout.to_string_lossy()
    }

    pub fn stderr_lossy(&self) -> String {
        self.stderr.to_string_lossy()
    }

    pub fn stdout(&self) -> &BoundedStream {
        &self.stdout
    }

    pub fn stderr(&self) -> &BoundedStream {
        &self.stderr
    }

    pub fn stdout_mut(&mut self) -> &mut BoundedStream {
        &mut self.stdout
    }

    pub fn stderr_mut(&mut self) -> &mut BoundedStream {
        &mut self.stderr
    }

    pub fn finish_stdout_spool(&mut self) -> Option<SpoolSummary> {
        self.stdout.finish_spool()
    }

    pub fn finish_stderr_spool(&mut self) -> Option<SpoolSummary> {
        self.stderr.finish_spool()
    }

    pub fn persist_stdout_artifact(
        &mut self,
        store: &ArtifactStore,
        id: ArtifactId,
        origin: ArtifactOrigin,
        retention: ArtifactRetention,
        scope: ArtifactScope,
        credential: &ScopeCredential,
    ) -> Result<Option<ArtifactMetadata>, ArtifactError> {
        self.stdout.persist_to_artifact_store(
            store,
            id,
            "text/plain",
            origin,
            retention,
            scope,
            credential,
        )
    }

    pub fn persist_stderr_artifact(
        &mut self,
        store: &ArtifactStore,
        id: ArtifactId,
        origin: ArtifactOrigin,
        retention: ArtifactRetention,
        scope: ArtifactScope,
        credential: &ScopeCredential,
    ) -> Result<Option<ArtifactMetadata>, ArtifactError> {
        self.stderr.persist_to_artifact_store(
            store,
            id,
            "text/plain",
            origin,
            retention,
            scope,
            credential,
        )
    }

    pub fn stdout_truncation_diag(
        &self,
        artifact_id: Option<ArtifactId>,
        fetchable: bool,
    ) -> Option<TruncationDiag> {
        self.stdout.truncation_diag(artifact_id, fetchable)
    }

    pub fn stderr_truncation_diag(
        &self,
        artifact_id: Option<ArtifactId>,
        fetchable: bool,
    ) -> Option<TruncationDiag> {
        self.stderr.truncation_diag(artifact_id, fetchable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_bounded_resident_buffer_with_disk_spool() {
        let max_resident_bytes = 20;
        let max_resident_lines = 10;
        let max_artifact_bytes = 10_000;

        let mut stream = BoundedStream::new_with_limits(
            StreamKind::Stdout,
            max_resident_bytes,
            max_resident_lines,
            max_artifact_bytes,
            true,
            None,
        );

        let data = b"0123456789ABCDEF0123456789extra_bytes_beyond_resident_buffer";
        stream.push(data);

        // Resident in-memory buffer is strictly bounded
        assert_eq!(stream.as_bytes().len(), max_resident_bytes);
        assert!(stream.is_truncated());
        assert_eq!(stream.observed_bytes(), data.len());

        // Disk spool preserved all data
        let summary = stream.finish_spool().unwrap();
        assert_eq!(summary.byte_length, data.len());
        assert!(!summary.truncated);

        let spooled_bytes = fs::read(&summary.path).unwrap();
        assert_eq!(spooled_bytes, data);

        let _ = fs::remove_file(&summary.path);
    }

    #[test]
    fn test_spool_direct_ingest_to_artifact_store() {
        let temp_dir = std::env::temp_dir().join(format!(
            "omp_stream_art_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let store = ArtifactStore::open(&temp_dir).unwrap();

        let mut stream =
            BoundedStream::new_with_limits(StreamKind::Stdout, 16, 10, 10_000, true, None);

        let payload =
            b"Large streamed content written directly to disk spool without RAM collection";
        stream.push(payload);

        let art_id = ArtifactId::mint();
        let cred = ScopeCredential::host();

        let meta = stream
            .persist_to_artifact_store(
                &store,
                art_id.clone(),
                "text/plain",
                ArtifactOrigin::System,
                ArtifactRetention::Transient,
                ArtifactScope::Global,
                &cred,
            )
            .unwrap()
            .unwrap();

        assert_eq!(meta.byte_length, payload.len());
        assert_eq!(meta.id, art_id);

        let read_blob = store.read_bounded(&meta, 1024, &cred).unwrap();
        assert_eq!(read_blob, payload);

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_accumulator_store_integration() {
        let temp_dir = std::env::temp_dir().join(format!(
            "omp_acc_test_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let store = Arc::new(ArtifactStore::open(&temp_dir).unwrap());
        let limits = LimitPolicy::default();
        let cred = ScopeCredential::host();

        let mut acc = BoundedStreamAccumulator::with_store(
            &limits,
            store.clone(),
            ArtifactOrigin::Job(omp_types::JobId::mint()),
            ArtifactScope::Global,
            cred.clone(),
        );

        let stdout_chunk = StreamChunk {
            kind: StreamKind::Stdout,
            data: b"Stdout content for job".to_vec(),
            timestamp_ms: 100,
        };
        let stderr_chunk = StreamChunk {
            kind: StreamKind::Stderr,
            data: b"Stderr content for job".to_vec(),
            timestamp_ms: 101,
        };

        acc.push_chunk(&stdout_chunk);
        acc.push_chunk(&stderr_chunk);

        let artifacts = acc.finish_artifacts().unwrap();
        assert_eq!(artifacts.len(), 2);

        let outputs = artifacts
            .iter()
            .map(|meta| store.read_bounded(meta, 1024, &cred).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            outputs,
            vec![
                b"Stdout content for job".to_vec(),
                b"Stderr content for job".to_vec()
            ]
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
