use crate::{SessionSnapshot, StateError, apply_patch};
use omp_types::{
    ActorId, BranchId, ElementId, ElementSnapshot, JOURNAL_FORMAT, JournalOffset, MAX_WIRE_BYTES,
    Patch, PatchAuthor, PatchOp, ProtocolVersion, SessionId, TypedValue,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const MAGIC: &[u8; 8] = b"OMP2J01\n";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    session_id: SessionId,
    format: String,
    version: ProtocolVersion,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JournalRecord {
    pub offset: u64,
    pub timestamp_ms: u64,
    pub session_id: SessionId,
    pub branch_id: BranchId,
    pub parent_offset: u64,
    pub patch: Patch,
    pub actor: PatchAuthor,
    pub protocol_version: ProtocolVersion,
    pub checksum: String,
    pub causal: Option<serde_json::Value>,
}
impl JournalRecord {
    fn checksum(&self) -> Result<String, StateError> {
        let mut value = self.clone();
        value.checksum.clear();
        Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecoveryDiagnostic {
    pub code: String,
    pub message: String,
    pub last_valid_offset: u64,
    pub valid_bytes: u64,
    pub damaged_bytes: u64,
    pub preserved_suffix: Option<PathBuf>,
}

/// Post-append observer receiving the durable record and resulting snapshot.
type JournalObserver = Box<dyn Fn(&JournalRecord, &SessionSnapshot) + Send + Sync>;

/// File owns the writer lock. Projections and branch indexes are replay-derived caches.
pub struct Journal {
    file: File,
    path: PathBuf,
    records: BTreeMap<u64, JournalRecord>,
    snapshot: SessionSnapshot,
    recovery: Option<RecoveryDiagnostic>,
    failed: bool,
    observer: Option<JournalObserver>,
}
fn write_frame(file: &mut File, bytes: &[u8]) -> Result<(), StateError> {
    if bytes.len() > MAX_WIRE_BYTES {
        return Err(StateError::Invalid("journal frame limit".into()));
    }
    file.write_all(&(bytes.len() as u32).to_le_bytes())?;
    file.write_all(bytes)?;
    file.write_all(&Sha256::digest(bytes))?;
    Ok(())
}
fn read_frame(file: &mut File) -> Result<Option<Vec<u8>>, StateError> {
    let mut len = [0; 4];
    let count = file.read(&mut len[..1])?;
    if count == 0 {
        return Ok(None);
    }
    file.read_exact(&mut len[1..])?;
    let len = u32::from_le_bytes(len) as usize;
    if len > MAX_WIRE_BYTES {
        return Err(StateError::Invalid("journal frame limit".into()));
    }
    let mut bytes = vec![0; len];
    file.read_exact(&mut bytes)?;
    let mut checksum = [0; 32];
    file.read_exact(&mut checksum)?;
    if Sha256::digest(&bytes)[..] != checksum {
        return Err(StateError::Invalid("journal frame checksum".into()));
    }
    Ok(Some(bytes))
}
impl Journal {
    pub fn create(path: impl AsRef<Path>, session_id: SessionId) -> Result<Self, StateError> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.try_lock()
            .map_err(|e| StateError::Invalid(format!("journal writer lock: {e}")))?;
        file.write_all(MAGIC)?;
        write_frame(
            &mut file,
            &serde_json::to_vec(&Header {
                session_id: session_id.clone(),
                format: JOURNAL_FORMAT.into(),
                version: ProtocolVersion::CURRENT,
            })?,
        )?;
        file.sync_all()?;
        Ok(Self {
            file,
            path,
            records: BTreeMap::new(),
            snapshot: SessionSnapshot::empty(session_id),
            recovery: None,
            failed: false,
            observer: None,
        })
    }
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StateError> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        file.try_lock()
            .map_err(|e| StateError::Invalid(format!("journal writer lock: {e}")))?;
        let mut magic = [0; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(StateError::Invalid("journal magic mismatch".into()));
        }
        let header: Header = serde_json::from_slice(
            &read_frame(&mut file)?
                .ok_or_else(|| StateError::Invalid("missing journal header".into()))?,
        )?;
        if header.format != JOURNAL_FORMAT || header.version != ProtocolVersion::CURRENT {
            return Err(StateError::Invalid("journal protocol mismatch".into()));
        }
        let mut journal = Self {
            file,
            path,
            records: BTreeMap::new(),
            snapshot: SessionSnapshot::empty(header.session_id),
            recovery: None,
            failed: false,
            observer: None,
        };
        loop {
            let start = journal.file.stream_position()?;
            let result = (|| -> Result<Option<JournalRecord>, StateError> {
                let Some(bytes) = read_frame(&mut journal.file)? else {
                    return Ok(None);
                };
                let record: JournalRecord = serde_json::from_slice(&bytes)?;
                journal.validate_record(&record)?;
                // Fast path: a contiguous record applies onto the running
                // snapshot, keeping replay linear. At a branch point
                // (fork/rewind/select) parent_offset references an ancestor,
                // so rematerialize from that offset instead. apply_patch is
                // transactional: a failed record leaves the snapshot at the
                // last valid state for the recovery path.
                if record.parent_offset == journal.snapshot.offset {
                    apply_patch(&mut journal.snapshot, &record.patch)?;
                } else {
                    let mut forked = journal.materialize(record.parent_offset)?;
                    apply_patch(&mut forked, &record.patch)?;
                    journal.snapshot = forked;
                }
                journal.snapshot.selected_branch = record.branch_id.clone();
                Ok(Some(record))
            })();
            match result {
                Ok(None) => break,
                Ok(Some(record)) => {
                    let offset = record.offset;
                    journal.records.insert(offset, record);
                }
                Err(error) => {
                    journal.recovery = Some(RecoveryDiagnostic {
                        code: "journal_damaged_suffix".into(),
                        message: error.to_string(),
                        last_valid_offset: journal.latest_offset(),
                        valid_bytes: start,
                        damaged_bytes: journal.file.metadata()?.len() - start,
                        preserved_suffix: None,
                    });
                    break;
                }
            }
        }
        Ok(journal)
    }
    fn validate_record(&self, record: &JournalRecord) -> Result<(), StateError> {
        if record.offset != self.latest_offset() + 1
            || record.parent_offset >= record.offset
            || record.patch.base_offset.0 != record.parent_offset
            || record.patch.result_offset.0 != record.offset
            || record.actor != record.patch.by
            || record.session_id != self.snapshot.session_id
            || record.protocol_version != ProtocolVersion::CURRENT
            || record.checksum != record.checksum()?
        {
            return Err(StateError::Invalid(
                "invalid journal record chain or checksum".into(),
            ));
        }
        Ok(())
    }
    pub fn snapshot(&self) -> &SessionSnapshot {
        &self.snapshot
    }
    /// Filesystem path backing this journal (sibling files such as token
    /// stores are derived from it).
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn set_observer(
        &mut self,
        observer: Option<JournalObserver>,
    ) {
        self.observer = observer;
    }
    pub fn records(&self) -> impl DoubleEndedIterator<Item = &JournalRecord> {
        self.records.values()
    }
    pub fn latest_offset(&self) -> u64 {
        self.records
            .last_key_value()
            .map_or(0, |(offset, _)| *offset)
    }
    pub fn next_offset(&self) -> JournalOffset {
        JournalOffset(self.latest_offset() + 1)
    }
    pub fn recovery(&self) -> Option<&RecoveryDiagnostic> {
        self.recovery.as_ref()
    }
    pub fn repair_suffix(&mut self) -> Result<Option<PathBuf>, StateError> {
        let Some(diagnostic) = &self.recovery else {
            return Ok(None);
        };
        let path = self
            .path
            .with_extension(format!("damaged-{}", SessionId::mint()));
        let mut suffix = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        self.file.seek(SeekFrom::Start(diagnostic.valid_bytes))?;
        std::io::copy(&mut self.file, &mut suffix)?;
        suffix.sync_all()?;
        self.file.set_len(diagnostic.valid_bytes)?;
        self.file.sync_all()?;
        self.recovery.as_mut().unwrap().preserved_suffix = Some(path.clone());
        self.recovery = None;
        self.failed = false;
        Ok(Some(path))
    }
    pub fn materialize(&self, offset: u64) -> Result<SessionSnapshot, StateError> {
        let mut chain = vec![];
        let mut cursor = offset;
        while cursor != 0 {
            let record = self
                .records
                .get(&cursor)
                .ok_or_else(|| StateError::Invalid(format!("unknown journal offset {cursor}")))?;
            chain.push(record);
            cursor = record.parent_offset;
        }
        let mut result = SessionSnapshot::empty(self.snapshot.session_id.clone());
        for record in chain.into_iter().rev() {
            apply_patch(&mut result, &record.patch)?;
            result.selected_branch = record.branch_id.clone();
        }
        Ok(result)
    }
    pub fn materialize_branch(&self, branch: &BranchId) -> Result<SessionSnapshot, StateError> {
        let offset = self
            .records
            .values()
            .rev()
            .find(|r| &r.branch_id == branch)
            .map(|r| r.offset);
        match offset {
            Some(offset) => self.materialize(offset),
            None if branch.as_str() == "main" => self.materialize(0),
            None => Err(StateError::Invalid("unknown branch".into())),
        }
    }
    pub fn append_patch(&mut self, patch: Patch) -> Result<u64, StateError> {
        self.append_on(patch, self.snapshot.selected_branch.clone())
    }
    fn append_on(&mut self, patch: Patch, branch: BranchId) -> Result<u64, StateError> {
        if self.failed || self.recovery.is_some() {
            return Err(StateError::WriteFailure);
        }
        if patch.result_offset != self.next_offset() {
            return Err(StateError::Invalid(
                "patch result must be next global offset".into(),
            ));
        }
        let previous = self.snapshot.offset;
        // Validation and mutation are transactional; no durable operation is acknowledged until sync.
        apply_patch(&mut self.snapshot, &patch)?;
        let result = (|| -> Result<JournalRecord, StateError> {
            let mut record = JournalRecord {
                offset: patch.result_offset.0,
                timestamp_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|e| StateError::Invalid(e.to_string()))?
                    .as_millis() as u64,
                session_id: self.snapshot.session_id.clone(),
                branch_id: branch.clone(),
                parent_offset: previous,
                actor: patch.by.clone(),
                patch,
                protocol_version: ProtocolVersion::CURRENT,
                checksum: String::new(),
                causal: None,
            };
            record.checksum = record.checksum()?;
            let bytes = serde_json::to_vec(&record)?;
            self.file.seek(SeekFrom::End(0))?;
            write_frame(&mut self.file, &bytes)?;
            self.file.sync_all()?;
            Ok(record)
        })();
        match result {
            Ok(record) => {
                let offset = record.offset;
                self.records.insert(offset, record);
                self.snapshot.selected_branch = branch;
                if let Some(observer) = &self.observer
                    && let Some(rec) = self.records.get(&offset) {
                        observer(rec, &self.snapshot);
                    }
                Ok(offset)
            }
            Err(error) => {
                self.snapshot = self.materialize(previous)?;
                self.failed = true;
                Err(error)
            }
        }
    }
    pub fn fork_at(&mut self, offset: u64) -> Result<BranchId, StateError> {
        let parent = self.materialize(offset)?;
        let branch = BranchId::mint();
        let mut element = ElementSnapshot::new(ElementId::mint(), "branch");
        element
            .attributes
            .insert("branch_id".into(), TypedValue::String(branch.to_string()));
        element.attributes.insert(
            "parent_offset".into(),
            TypedValue::Json(serde_json::json!(offset)),
        );
        let index = parent.children(parent.container("branches")).count() as u32;
        let patch = Patch {
            base_offset: JournalOffset(offset),
            result_offset: self.next_offset(),
            by: ActorId::new("host").unwrap().into(),
            reason: "fork branch".into(),
            ops: vec![PatchOp::Create {
                parent: parent.container("branches").clone(),
                index,
                element,
            }],
        };
        let old = std::mem::replace(&mut self.snapshot, parent);
        if let Err(error) = self.append_on(patch, branch.clone()) {
            self.snapshot = old;
            return Err(error);
        }
        Ok(branch)
    }
    pub fn rewind_to(&mut self, offset: u64) -> Result<BranchId, StateError> {
        self.fork_at(offset)
    }
    pub fn select_branch(&mut self, branch: &BranchId) -> Result<u64, StateError> {
        let target = self.materialize_branch(branch)?;
        let patch = Patch {
            base_offset: JournalOffset(target.offset),
            result_offset: self.next_offset(),
            by: ActorId::new("host").unwrap().into(),
            reason: "select branch".into(),
            ops: vec![],
        };
        let old = std::mem::replace(&mut self.snapshot, target);
        match self.append_on(patch, branch.clone()) {
            Ok(offset) => Ok(offset),
            Err(error) => {
                self.snapshot = old;
                Err(error)
            }
        }
    }
    pub fn resume_latest(&self) -> SessionSnapshot {
        self.snapshot.clone()
    }
    pub fn branch_ancestry(&self, offset: u64) -> Result<Vec<u64>, StateError> {
        let mut offsets = vec![];
        let mut cursor = offset;
        while cursor != 0 {
            offsets.push(cursor);
            cursor = self
                .records
                .get(&cursor)
                .ok_or_else(|| StateError::Invalid("unknown offset".into()))?
                .parent_offset;
        }
        offsets.reverse();
        Ok(offsets)
    }
    pub fn inspect(&self, offset: u64) -> Result<serde_json::Value, StateError> {
        let snapshot = self.materialize(offset)?;
        Ok(
            serde_json::json!({"offset":offset,"branch":snapshot.current_branch(),"ancestry":self.branch_ancestry(offset)?,"tree":snapshot.inspect_xml(),"turn_count":snapshot.turn_count(),"active_tools":snapshot.active_tool_roster().map(|e|&e.id).collect::<Vec<_>>(),"active_jobs":snapshot.active_jobs().map(|e|&e.id).collect::<Vec<_>>(),"snapshot":snapshot}),
        )
    }
}
