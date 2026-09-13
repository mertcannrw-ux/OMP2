use crate::SessionHost;
use omp_runtime::WorkspaceView;
use omp_state::Journal;
use omp_tools::ToolExecutionResult;
use omp_types::{
    ActorId, ElementId, ElementSnapshot, JournalOffset, Patch, PatchOp, SessionId, StructuredError,
    TypedValue,
};
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};

pub(crate) struct ChildHandle {
    pub cancel: Arc<AtomicBool>,
    pub result: mpsc::Receiver<Result<serde_json::Value, StructuredError>>,
    pub thread: Option<std::thread::JoinHandle<()>>,
}
impl Drop for ChildHandle {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}

impl SessionHost {
    pub(crate) fn spawn_children(
        &mut self,
        journal: &mut Journal,
        input: &serde_json::Value,
    ) -> Result<ToolExecutionResult, StructuredError> {
        let context = input["context"].as_str().ok_or_else(|| {
            StructuredError::new("agent_input", "Agent context is required", false)
        })?;
        // Bound child input context (64 KiB) so an unbounded parent context
        // cannot blow up child journals or thread stacks.
        const MAX_CHILD_CONTEXT_BYTES: usize = 64 * 1024;
        if context.len() > MAX_CHILD_CONTEXT_BYTES {
            return Err(StructuredError::new(
                "agent_input_too_large",
                format!(
                    "Agent context is {} bytes, exceeding the {} byte child limit",
                    context.len(),
                    MAX_CHILD_CONTEXT_BYTES
                ),
                false,
            ));
        }
        let tasks = input["tasks"].as_array().ok_or_else(|| {
            StructuredError::new("agent_input", "Agent tasks are required", false)
        })?;
        let max = self
            .convars
            .get_typed::<i64>("job_max_concurrency")
            .unwrap_or(4)
            .clamp(1, 32) as usize;
        if tasks.is_empty() || tasks.len() + self.children.len() > max {
            return Err(StructuredError::new(
                "agent_limit",
                "Batch exceeds available child concurrency",
                false,
            ));
        }
        for task in tasks {
            if task["task"]
                .as_str()
                .is_none_or(|value| value.trim().is_empty() || value.len() > 30000)
            {
                return Err(StructuredError::new(
                    "agent_input",
                    "Each child requires a bounded nonempty task",
                    false,
                ));
            }
        }
        let mut ids = Vec::new();
        for task in tasks {
            let actor = ActorId::mint();
            let element = ElementId::new(format!("actor-{actor}"))?;
            let child_id = SessionId::mint();
            let view = WorkspaceView::allocate(
                &self.workspace,
                WorkspaceView::default_views_root(&self.workspace),
                child_id.clone(),
            )?;
            let child_dir = self.workspace.join(".omp/children");
            std::fs::create_dir_all(&child_dir)
                .map_err(|error| StructuredError::new("child_journal", error.to_string(), false))?;
            let child_path = child_dir.join(format!("{child_id}.journal"));
            let mut child_journal =
                Journal::create(&child_path, child_id).map_err(|error| error.structured())?;
            let mut child = self.seed_child_host(
                journal.snapshot(),
                &mut child_journal,
                view.isolated_path.clone(),
                actor.clone(),
                task["agent"].as_str(),
            )?;
            child.provider.endpoint = self.provider.endpoint.clone();
            child.provider.timeout_secs = child.provider.timeout_secs.min(30);
            let cancellation = Arc::new(AtomicBool::new(false));
            child.cancellation = Some(cancellation.clone());
            let mut node = ElementSnapshot::new(element.clone(), "subagent");
            node.attributes
                .insert("actor_id".into(), TypedValue::String(actor.to_string()));
            node.attributes
                .insert("status".into(), TypedValue::String("running".into()));
            node.attributes
                .insert("parent".into(), TypedValue::String(self.owner.to_string()));
            node.payload = Some(
                serde_json::json!({"workspace":view,"journal":child_path,"prompt":task["task"],"context":context,"convars":child_journal.snapshot().session_globals()}),
            );
            journal
                .append_patch(Patch {
                    base_offset: JournalOffset(journal.snapshot().offset),
                    result_offset: journal.next_offset(),
                    by: self.owner.clone().into(),
                    reason: "spawn isolated child host".into(),
                    ops: vec![PatchOp::Create {
                        parent: journal.snapshot().container("actors").clone(),
                        index: journal
                            .snapshot()
                            .children(journal.snapshot().container("actors"))
                            .count() as u32,
                        element: node,
                    }],
                })
                .map_err(|error| error.structured())?;
            let prompt = format!("{context}\n\n{}", task["task"].as_str().unwrap());
            let (tx, rx) = mpsc::sync_channel(1);
            let thread = std::thread::Builder::new().name(format!("child-{actor}")).spawn(move || {
                let result = child.run_turn(&mut child_journal, &prompt);
                let shutdown = child.shutdown(&mut child_journal);
                let text = child_journal.snapshot().get_visible_body().filter(|node| node.kind == "assistant").map(|node| node.text.as_str()).collect::<Vec<_>>().join("\n");
                let diff = view.compute_diff();
                let error = result.err().or_else(|| shutdown.err()).or_else(|| diff.as_ref().err().cloned());
                let status = match error.as_ref().map(|error| error.code.as_str()) { Some("cancelled") => "cancelled", Some(_) => "failed", None => "succeeded" };
                let payload = serde_json::json!({"status":status,"result":text,"error":error,"diff":diff.ok(),"workspace":view,"journal":child_path,"offset":child_journal.snapshot().offset});
                let _ = tx.send(Ok(payload));
            }).map_err(|error| StructuredError::new("child_spawn", error.to_string(), false))?;
            self.children.insert(
                element.clone(),
                ChildHandle {
                    cancel: cancellation,
                    result: rx,
                    thread: Some(thread),
                },
            );
            ids.push(format!("agent://{element}"));
        }
        Ok(ToolExecutionResult::success(ids.join("\n")))
    }

    pub fn poll_children(&mut self, journal: &mut Journal) -> Result<(), StructuredError> {
        let mut completed = BTreeMap::new();
        for (id, child) in &self.children {
            if journal.snapshot().element(id).is_none() {
                child.cancel.store(true, Ordering::Release);
            }
            match child.result.try_recv() {
                Ok(result) => {
                    completed.insert(id.clone(), result);
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    completed.insert(
                        id.clone(),
                        Err(StructuredError::new(
                            "child_disconnected",
                            "Child host ended without a result",
                            false,
                        )),
                    );
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        for (id, result) in completed {
            if let Some(mut child) = self.children.remove(&id)
                && let Some(thread) = child.thread.take() {
                    let _ = thread.join();
                }
            if journal.snapshot().element(&id).is_none() {
                continue;
            }
            let payload = match result {
                Ok(value) => value,
                Err(error) => serde_json::json!({"status":"failed","error":error}),
            };
            // Settle must stay atomic: truncate the result text so an
            // oversized child transcript cannot fail the whole patch after
            // the child was already removed from the table.
            let text = payload["result"].as_str().unwrap_or("");
            let text = truncate_settle_text(text, 64 * 1024);
            // Redact host-absolute paths: the persisted payload keeps only the
            // view id and offset, never the full `WorkspaceView` (with
            // baseline hashes) or absolute journal path.
            let redacted = serde_json::json!({
                "status": payload["status"],
                "result": text,
                "error": payload["error"],
                "diff": payload["diff"],
                "offset": payload["offset"],
            });
            journal
                .append_patch(Patch {
                    base_offset: JournalOffset(journal.snapshot().offset),
                    result_offset: journal.next_offset(),
                    by: self.owner.clone().into(),
                    reason: "settle child host".into(),
                    ops: vec![
                        PatchOp::SetAttribute {
                            element: id.clone(),
                            name: "status".into(),
                            value: TypedValue::String(
                                payload["status"].as_str().unwrap_or("failed").into(),
                            ),
                        },
                        PatchOp::ReplaceText {
                            element: id.clone(),
                            text,
                        },
                        PatchOp::ReplacePayload {
                            element: id,
                            payload: redacted,
                        },
                    ],
                })
                .map_err(|error| error.structured())?;
        }
        Ok(())
    }

    pub fn shutdown(&mut self, journal: &mut Journal) -> Result<(), StructuredError> {
        for child in self.children.values() {
            child.cancel.store(true, Ordering::Release);
        }
        self.tool_host
            .shutdown_jobs(journal)
            .map_err(|error| StructuredError::new("shutdown", error.to_string(), false))?;
        while !self.children.is_empty() {
            self.poll_children(journal)?;
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(())
    }
}

/// Truncate settle text on a UTF-8 boundary so oversized child transcripts
/// keep the settle patch atomic.
fn truncate_settle_text(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut cut = max_bytes;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    text[..cut].to_string()
}
