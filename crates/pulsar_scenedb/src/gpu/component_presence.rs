//! Per-component GPU presence for `World`-mirrored components.
//!
//! Entity generations answer "is this still the same entity?"; they cannot
//! answer "does that entity still have component `T`?" because removing one
//! component does not (and must not) change the entity generation.  Each
//! GPU-bearing component therefore owns one of these `u32` row buffers:
//! `1` means the component is present, `0` is the explicit tombstone.
//!
//! Consumers must check both the entity generation and this presence value
//! before interpreting a mirrored component row.  The value row is also
//! zeroed on removal as hygiene, but zero bytes are not themselves a generic
//! absence sentinel (zero can be a perfectly valid mesh/material index).

use crate::gpu::{
    CapacityError, DirtyTrackedReallocationPolicy, DirtyTrackedSceneBuffer, SyncStats,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

/// A growable `u32` presence column plus CPU transition state.
///
/// The transition state prevents ordinary in-place component updates from
/// re-uploading an unchanged `1` every frame.  It is separate from the
/// dirty-tracked buffer's private shadow deliberately: presence transitions
/// are also what define `Once`'s one-time handoff lifetime.
pub struct ComponentPresenceBuffer {
    buf: DirtyTrackedSceneBuffer<u32>,
    rows: RwLock<Vec<AtomicBool>>,
}

impl ComponentPresenceBuffer {
    pub fn new(device: Arc<wgpu::Device>, label: &str, initial_capacity: u32) -> Self {
        Self::new_with_reallocation_policy(
            device,
            label,
            initial_capacity,
            DirtyTrackedReallocationPolicy::GpuCopy,
        )
    }

    pub fn new_with_reallocation_policy(
        device: Arc<wgpu::Device>,
        label: &str,
        initial_capacity: u32,
        reallocation_policy: DirtyTrackedReallocationPolicy,
    ) -> Self {
        Self {
            buf: DirtyTrackedSceneBuffer::new_with_reallocation_policy(
                device,
                label,
                initial_capacity,
                reallocation_policy,
            ),
            rows: RwLock::new(
                (0..initial_capacity)
                    .map(|_| AtomicBool::new(false))
                    .collect(),
            ),
        }
    }

    /// Marks `row` present. Returns `true` only for an absent -> present
    /// transition (the point a `Once` field performs its handoff).
    pub fn mark_present(&self, row: u32) -> bool {
        let idx = row as usize;
        {
            let rows = self
                .rows
                .read()
                .expect("ComponentPresenceBuffer lock poisoned");
            if idx < rows.len() {
                if rows[idx].swap(true, Ordering::Relaxed) {
                    return false;
                }
                self.buf.mark_dirty(row, 1);
                return true;
            }
        }

        let mut rows = self
            .rows
            .write()
            .expect("ComponentPresenceBuffer lock poisoned");
        if idx >= rows.len() {
            rows.resize_with(idx + 1, || AtomicBool::new(false));
        }
        if rows[idx].swap(true, Ordering::Relaxed) {
            return false;
        }
        self.buf.mark_dirty(row, 1);
        true
    }

    /// Marks `row` absent. Returns `true` only for a present -> absent
    /// transition. Rows never observed as present remain a no-op.
    pub fn mark_absent(&self, row: u32) -> bool {
        let idx = row as usize;
        let rows = self
            .rows
            .read()
            .expect("ComponentPresenceBuffer lock poisoned");
        let Some(flag) = rows.get(idx) else {
            return false;
        };
        if !flag.swap(false, Ordering::Relaxed) {
            return false;
        }
        self.buf.mark_dirty(row, 0);
        true
    }

    pub fn flush(&self, queue: &wgpu::Queue) -> SyncStats {
        self.buf.flush(queue)
    }

    pub fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        // Check the bounded GPU allocation before attempting an equivalently
        // huge host-side presence vector. Invalid reservations must remain a
        // catchable CapacityError, not become allocator pressure/hangs.
        self.buf.reserve(queue, capacity)?;
        {
            let mut rows = self
                .rows
                .write()
                .expect("ComponentPresenceBuffer lock poisoned");
            if rows.len() < capacity as usize {
                rows.resize_with(capacity as usize, || AtomicBool::new(false));
            }
        }
        Ok(())
    }

    pub fn shrink_to_fit(
        &self,
        queue: &wgpu::Queue,
        highest_live_row: u32,
        slack_factor: f32,
    ) -> bool {
        let target = (((highest_live_row as u64 + 1) as f64 * slack_factor.max(1.0) as f64).ceil()
            as u64)
            .min(u32::MAX as u64) as usize;
        {
            let mut rows = self
                .rows
                .write()
                .expect("ComponentPresenceBuffer lock poisoned");
            if target < rows.len() {
                rows.truncate(target);
                rows.shrink_to_fit();
            }
        }
        self.buf
            .shrink_to_fit(queue, highest_live_row, slack_factor)
    }

    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        self.buf.with_buffer(f);
    }

    /// Atomically snapshots the presence buffer and its allocation epoch.
    pub fn buffer_snapshot(&self) -> (wgpu::Buffer, u64) {
        self.buf.buffer_snapshot()
    }

    pub fn epoch(&self) -> u64 {
        self.buf.epoch()
    }
}
