use crate::error::ServerError;
use crate::role::ActorRole;
use omp_state::SessionSnapshot;
use omp_types::{ActorId, JournalOffset, Patch};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const MAX_SUBSCRIBE_CAPACITY: usize = 4096;
pub const MIN_SUBSCRIBE_CAPACITY: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReplicationMessage {
    Resync {
        snapshot: SessionSnapshot,
        offset: JournalOffset,
    },
    Patch {
        patch: Patch,
    },
    Presentation {
        view_id: String,
        payload: serde_json::Value,
    },
    Heartbeat {
        timestamp_epoch_ms: u64,
        latest_offset: JournalOffset,
    },
    Lagged {
        dropped_presentation_frames: usize,
    },
}

#[derive(Debug)]
pub struct SubscriberQueue {
    pub actor_id: ActorId,
    pub role: ActorRole,
    pub last_acknowledged_offset: JournalOffset,
    pub capacity: usize,
    pub queue: VecDeque<ReplicationMessage>,
    pub dropped_presentation_count: usize,
    pub needs_resync: bool,
}

impl SubscriberQueue {
    pub fn new(
        actor_id: ActorId,
        role: ActorRole,
        last_offset: JournalOffset,
        capacity: usize,
    ) -> Self {
        let capacity = capacity.clamp(MIN_SUBSCRIBE_CAPACITY, MAX_SUBSCRIBE_CAPACITY);
        Self {
            actor_id,
            role,
            last_acknowledged_offset: last_offset,
            capacity,
            queue: VecDeque::new(),
            dropped_presentation_count: 0,
            needs_resync: false,
        }
    }

    /// Enqueues a replication message with proper lag handling and frame coalescing.
    ///
    /// Presentation frames are coalesced/dropped on overflow.
    /// Journal overflow requires a snapshot resync for every role before more
    /// patches can be delivered; a slow client cannot stall authoritative writes.
    pub fn push(&mut self, message: ReplicationMessage) -> Result<(), ServerError> {
        match message {
            ReplicationMessage::Presentation { view_id, payload } => {
                // Coalesce intermediate presentation frames with existing frame for same view
                for existing in self.queue.iter_mut().rev() {
                    if let ReplicationMessage::Presentation {
                        view_id: v_id,
                        payload: p,
                    } = existing
                        && *v_id == view_id {
                            *p = payload;
                            self.dropped_presentation_count += 1;
                            return Ok(());
                        }
                }
                // If queue is full and no coalescing target found, drop intermediate presentation frame
                if self.queue.len() >= self.capacity {
                    self.dropped_presentation_count += 1;
                    return Ok(());
                }
                self.queue
                    .push_back(ReplicationMessage::Presentation { view_id, payload });
                Ok(())
            }
            ReplicationMessage::Patch { patch } => {
                if self.needs_resync {
                    // Actor is already pending resync, do not queue further patches
                    return Ok(());
                }

                if self.queue.len() >= self.capacity {
                    self.needs_resync = true;
                    self.queue.clear();
                    return Err(ServerError::SubscriberQueueFull(self.actor_id.to_string()));
                }

                self.last_acknowledged_offset = patch.result_offset;
                self.queue.push_back(ReplicationMessage::Patch { patch });
                Ok(())
            }
            ReplicationMessage::Resync { snapshot, offset } => {
                // Resync clears old pending patches and resets state
                self.queue.clear();
                self.last_acknowledged_offset = offset;
                self.needs_resync = false;
                self.queue
                    .push_back(ReplicationMessage::Resync { snapshot, offset });
                Ok(())
            }
            other => {
                if self.queue.len() < self.capacity {
                    self.queue.push_back(other);
                }
                Ok(())
            }
        }
    }

    pub fn pop(&mut self) -> Option<ReplicationMessage> {
        self.queue.pop_front()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn take_dropped_presentation_count(&mut self) -> usize {
        let count = self.dropped_presentation_count;
        self.dropped_presentation_count = 0;
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omp_types::{ActorId, JournalOffset, Patch, SessionId};

    fn dummy_patch(base: u64, result: u64) -> Patch {
        Patch {
            base_offset: JournalOffset(base),
            result_offset: JournalOffset(result),
            by: ActorId::new("tester").unwrap().into(),
            reason: "test patch".into(),
            ops: vec![],
        }
    }

    #[test]
    fn presentation_coalesces_in_place() {
        let mut q = SubscriberQueue::new(
            ActorId::new("sub-1").unwrap(),
            ActorRole::Spectator,
            JournalOffset(0),
            16,
        );

        q.push(ReplicationMessage::Presentation {
            view_id: "tui".into(),
            payload: serde_json::json!({"frame": 1}),
        })
        .unwrap();
        assert_eq!(q.len(), 1);
        assert_eq!(q.dropped_presentation_count, 0);

        // Subsequent frame for same view_id coalesces in-place
        q.push(ReplicationMessage::Presentation {
            view_id: "tui".into(),
            payload: serde_json::json!({"frame": 2}),
        })
        .unwrap();
        assert_eq!(q.len(), 1);
        assert_eq!(q.dropped_presentation_count, 1);

        if let Some(ReplicationMessage::Presentation { view_id, payload }) = q.pop() {
            assert_eq!(view_id, "tui");
            assert_eq!(payload["frame"], 2);
        } else {
            panic!("expected presentation frame");
        }
    }

    #[test]
    fn presentation_dropped_when_full_without_error() {
        let mut q = SubscriberQueue::new(
            ActorId::new("sub-1").unwrap(),
            ActorRole::Spectator,
            JournalOffset(0),
            16,
        );

        for i in 0..16 {
            q.push(ReplicationMessage::Presentation {
                view_id: format!("view-{i}"),
                payload: serde_json::json!({"idx": i}),
            })
            .unwrap();
        }
        assert_eq!(q.len(), 16);

        // 17th frame for new view dropped
        q.push(ReplicationMessage::Presentation {
            view_id: "view-overflow".into(),
            payload: serde_json::json!({"idx": 99}),
        })
        .unwrap();
        assert_eq!(q.len(), 16);
        assert_eq!(q.take_dropped_presentation_count(), 1);
    }

    #[test]
    fn patches_never_dropped_and_maintain_order() {
        let mut q = SubscriberQueue::new(
            ActorId::new("sub-1").unwrap(),
            ActorRole::InteractiveDriver,
            JournalOffset(0),
            16,
        );

        for i in 1..=10 {
            q.push(ReplicationMessage::Patch {
                patch: dummy_patch(i - 1, i),
            })
            .unwrap();
        }
        assert_eq!(q.len(), 10);
        assert_eq!(q.last_acknowledged_offset, JournalOffset(10));

        for i in 1..=10 {
            match q.pop() {
                Some(ReplicationMessage::Patch { patch }) => {
                    assert_eq!(patch.result_offset, JournalOffset(i));
                }
                other => panic!("expected patch {i}, got {other:?}"),
            }
        }
    }

    #[test]
    fn spectator_overflow_clears_queue_and_marks_resync() {
        let mut q = SubscriberQueue::new(
            ActorId::new("spec-1").unwrap(),
            ActorRole::Spectator,
            JournalOffset(0),
            16,
        );

        for i in 1..=16 {
            q.push(ReplicationMessage::Patch {
                patch: dummy_patch(i - 1, i),
            })
            .unwrap();
        }
        assert_eq!(q.len(), 16);

        // 17th patch causes overflow
        let err = q
            .push(ReplicationMessage::Patch {
                patch: dummy_patch(16, 17),
            })
            .unwrap_err();

        assert!(matches!(err, ServerError::SubscriberQueueFull(_)));
        assert!(q.needs_resync);
        assert_eq!(q.len(), 0); // Cleared to prevent spectator unbounded memory
    }

    #[test]
    fn resync_resets_queue_state() {
        let mut q = SubscriberQueue::new(
            ActorId::new("spec-1").unwrap(),
            ActorRole::Spectator,
            JournalOffset(0),
            16,
        );
        q.needs_resync = true;

        let snapshot = SessionSnapshot::empty(SessionId::mint());
        q.push(ReplicationMessage::Resync {
            snapshot,
            offset: JournalOffset(42),
        })
        .unwrap();

        assert!(!q.needs_resync);
        assert_eq!(q.len(), 1);
        assert_eq!(q.last_acknowledged_offset, JournalOffset(42));
    }

    #[test]
    fn controller_overflow_requires_resync_before_more_patches() {
        let mut q = SubscriberQueue::new(
            ActorId::new("ctrl-1").unwrap(),
            ActorRole::Controller,
            JournalOffset(0),
            16,
        );

        for i in 1..=16 {
            q.push(ReplicationMessage::Patch {
                patch: dummy_patch(i - 1, i),
            })
            .unwrap();
        }
        assert_eq!(q.len(), 16);

        let err = q
            .push(ReplicationMessage::Patch {
                patch: dummy_patch(16, 17),
            })
            .unwrap_err();

        assert!(matches!(err, ServerError::SubscriberQueueFull(_)));
        assert!(q.needs_resync);
        q.push(ReplicationMessage::Patch {
            patch: dummy_patch(17, 18),
        })
        .unwrap();
        assert!(
            q.pop().is_none(),
            "No post-gap patch may escape before resync"
        );
    }

    #[test]
    fn subscribe_capacity_is_clamped_and_lazy() {
        let q = SubscriberQueue::new(
            ActorId::new("sub-cap").unwrap(),
            ActorRole::Spectator,
            JournalOffset(0),
            usize::MAX,
        );
        assert_eq!(q.capacity, MAX_SUBSCRIBE_CAPACITY);
        assert!(q.queue.capacity() < 1024);
    }
}
