//! Deferred one-time handoff buffers for `#[gpu(mirror = Once)]` World
//! fields.
//!
//! Unlike [`super::DirtyTrackedSceneBuffer`], this path does not retain a
//! full CPU shadow of every GPU row. A handoff appends `(row, value)` to a
//! transient queue; flush stable-sorts and collapses duplicate rows so the
//! last lifecycle event wins, then uses the same direct-vs-GPU-scatter
//! strategy as dirty tracking. The queue is cleared after the flush.
//!
//! "Once" means once per *component-presence lifetime*: first insertion
//! hands the value off, ordinary in-place inserts do not, removal writes a
//! tombstone, and a later re-insertion starts a new lifetime and hands off
//! again.

use crate::gpu::scatter_write::ScatterWritePipeline;
use crate::gpu::{CapacityError, DynamicGpuBuffer, SyncStats};
use crate::page::Pod;
use std::sync::{Arc, Mutex, OnceLock};

struct OnceState<T: Pod> {
    buf: DynamicGpuBuffer<T>,
    pending: Vec<(u32, T)>,
    upload_indices: Vec<u32>,
    upload_values: Vec<T>,
    upload_words: Vec<u32>,
    scatter_scratch: Option<OnceScatterScratch>,
}

struct OnceScatterScratch {
    indices: DynamicGpuBuffer<u32>,
    values: DynamicGpuBuffer<u32>,
}

/// Type-erased storage used by [`super::SceneGpuStore`].
pub trait OnceGpuBufferDispatch: Send + Sync {
    fn queue_handoff_bytes(&self, row: u32, data: &[u8]);
    fn flush(&self, queue: &wgpu::Queue) -> SyncStats;
    fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError>;
    fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool;
    fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer));
    /// Atomically snapshots the physical buffer handle and the epoch which
    /// identifies that allocation. Callers must not fetch these separately:
    /// a concurrent grow between two calls could otherwise pair a new handle
    /// with an old epoch (or vice versa).
    fn buffer_snapshot(&self) -> (wgpu::Buffer, u64);
    fn epoch(&self) -> u64;
    fn capacity(&self) -> u32;
    fn pending_count(&self) -> usize;
}

pub struct OnceSceneBuffer<T: Pod> {
    device: Arc<wgpu::Device>,
    /// Most Once workloads coalesce into a handful of direct writes. Avoid
    /// requiring shader-module/compute support until a flush genuinely uses
    /// the many-run scatter path.
    scatter: OnceLock<ScatterWritePipeline>,
    label: String,
    state: Mutex<OnceState<T>>,
}

impl<T: Pod + Send + Sync + 'static> OnceSceneBuffer<T> {
    const SCATTER_RUN_THRESHOLD: u32 = 4;

    pub fn new(device: Arc<wgpu::Device>, label: &str, initial_capacity: u32) -> Self {
        assert_eq!(
            std::mem::size_of::<T>() % 4,
            0,
            "OnceSceneBuffer requires size_of::<T>() to be a multiple of 4 bytes"
        );
        let state = OnceState {
            buf: DynamicGpuBuffer::new(&device, label, initial_capacity),
            pending: Vec::new(),
            upload_indices: Vec::new(),
            upload_values: Vec::new(),
            upload_words: Vec::new(),
            scatter_scratch: None,
        };
        Self {
            scatter: OnceLock::new(),
            label: label.to_owned(),
            device,
            state: Mutex::new(state),
        }
    }

    /// Queues one lifecycle handoff. Duplicate rows are intentionally kept
    /// until flush; stable collapse there preserves the last event.
    pub fn queue_handoff(&self, row: u32, value: T) {
        self.state
            .lock()
            .expect("OnceSceneBuffer lock poisoned")
            .pending
            .push((row, value));
    }

    pub fn flush(&self, queue: &wgpu::Queue) -> SyncStats {
        let mut state = self.state.lock().expect("OnceSceneBuffer lock poisoned");
        if state.pending.is_empty() {
            return SyncStats {
                ranges: 0,
                bytes: 0,
            };
        }

        // Stable sort is load-bearing: for duplicate rows, insertion order
        // is lifecycle order and the final entry must win.
        state.pending.sort_by_key(|&(row, _)| row);
        let mut out = 0usize;
        for read in 0..state.pending.len() {
            let entry = state.pending[read];
            if out > 0 && state.pending[out - 1].0 == entry.0 {
                state.pending[out - 1] = entry;
            } else {
                state.pending[out] = entry;
                out += 1;
            }
        }
        state.pending.truncate(out);

        let required = state.pending.last().unwrap().0.saturating_add(1);
        state
            .buf
            .ensure_capacity(&self.device, queue, required)
            .expect("OnceSceneBuffer has no max_capacity -- growth cannot fail");

        let pending = std::mem::take(&mut state.pending);
        state.upload_indices.clear();
        state.upload_values.clear();
        state.upload_indices.reserve(pending.len());
        state.upload_values.reserve(pending.len());
        let mut run_count = 0u32;
        let mut run_end = 0u32;
        for &(row, value) in &pending {
            if state.upload_indices.is_empty() || row != run_end {
                run_count += 1;
            }
            run_end = row + 1;
            state.upload_indices.push(row);
            state.upload_values.push(value);
        }

        // Move transient arrays out while consuming them. This keeps reads of
        // staging data disjoint from mutations of the GPU-buffer fields
        // behind `MutexGuard` and preserves the allocations for reuse below.
        let upload_indices = std::mem::take(&mut state.upload_indices);
        let upload_values = std::mem::take(&mut state.upload_values);
        let mut upload_words = std::mem::take(&mut state.upload_words);

        let count = upload_indices.len() as u32;
        let stats = if run_count <= Self::SCATTER_RUN_THRESHOLD {
            let mut stats = SyncStats {
                ranges: 0,
                bytes: 0,
            };
            let mut begin = 0usize;
            while begin < upload_indices.len() {
                let start_row = upload_indices[begin];
                let mut end = begin + 1;
                while end < upload_indices.len()
                    && upload_indices[end] == upload_indices[end - 1] + 1
                {
                    end += 1;
                }
                state
                    .buf
                    .write(queue, start_row, &upload_values[begin..end]);
                stats.ranges += 1;
                stats.bytes += ((end - begin) * std::mem::size_of::<T>()) as u64;
                begin = end;
            }
            stats
        } else {
            let words_per_element = (std::mem::size_of::<T>() / 4) as u32;
            if state.scatter_scratch.is_none() {
                state.scatter_scratch = Some(OnceScatterScratch {
                    indices: DynamicGpuBuffer::new(
                        &self.device,
                        &format!("{}-once-scatter-indices", self.label),
                        1,
                    ),
                    values: DynamicGpuBuffer::new(
                        &self.device,
                        &format!("{}-once-scatter-values", self.label),
                        1,
                    ),
                });
            }
            let OnceState {
                buf,
                scatter_scratch,
                ..
            } = &mut *state;
            let scratch = scatter_scratch
                .as_mut()
                .expect("Once scatter scratch initialized");
            scratch
                .indices
                .ensure_capacity(&self.device, queue, count)
                .expect("Once scatter-index scratch is unbounded");
            scratch
                .values
                .ensure_capacity(&self.device, queue, count * words_per_element)
                .expect("Once scatter-value scratch is unbounded");
            scratch.indices.write(queue, 0, &upload_indices);

            upload_words.clear();
            upload_words.reserve(count as usize * words_per_element as usize);
            for value in &upload_values {
                // SAFETY: `T: Pod`; the World GPU path additionally requires
                // whole-u32 sizing, asserted in `new`.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        value as *const T as *const u8,
                        std::mem::size_of::<T>(),
                    )
                };
                upload_words.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|word| u32::from_ne_bytes(word.try_into().unwrap())),
                );
            }
            scratch.values.write(queue, 0, &upload_words);
            self.scatter
                .get_or_init(|| ScatterWritePipeline::new(&self.device))
                .dispatch(
                &self.device,
                queue,
                scratch.indices.buffer(),
                scratch.values.buffer(),
                buf.buffer(),
                words_per_element,
                count,
            );
            SyncStats {
                ranges: 1,
                bytes: count as u64 * std::mem::size_of::<T>() as u64,
            }
        };

        state.pending = pending;
        state.pending.clear();
        state.upload_indices = upload_indices;
        state.upload_indices.clear();
        state.upload_values = upload_values;
        state.upload_values.clear();
        state.upload_words = upload_words;
        state.upload_words.clear();
        stats
    }

    pub fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        self.state
            .lock()
            .expect("OnceSceneBuffer lock poisoned")
            .buf
            .reserve(&self.device, queue, capacity)
    }

    pub fn shrink_to_fit(
        &self,
        queue: &wgpu::Queue,
        highest_live_row: u32,
        slack_factor: f32,
    ) -> bool {
        let mut state = self.state.lock().expect("OnceSceneBuffer lock poisoned");
        let shrank = state
            .buf
            .shrink_to_fit(&self.device, queue, highest_live_row, slack_factor);
        // These are transient staging queues, not row shadows. A natural
        // shrink boundary is also the right time to release their high-water
        // allocations.
        state.pending.shrink_to_fit();
        state.upload_indices.shrink_to_fit();
        state.upload_values.shrink_to_fit();
        state.upload_words.shrink_to_fit();
        shrank
    }

    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        let state = self.state.lock().expect("OnceSceneBuffer lock poisoned");
        f(state.buf.buffer());
    }

    pub fn buffer_snapshot(&self) -> (wgpu::Buffer, u64) {
        let state = self.state.lock().expect("OnceSceneBuffer lock poisoned");
        (state.buf.buffer().clone(), state.buf.epoch())
    }

    pub fn epoch(&self) -> u64 {
        self.state
            .lock()
            .expect("OnceSceneBuffer lock poisoned")
            .buf
            .epoch()
    }

    pub fn capacity(&self) -> u32 {
        self.state
            .lock()
            .expect("OnceSceneBuffer lock poisoned")
            .buf
            .capacity()
    }

    pub fn pending_count(&self) -> usize {
        self.state
            .lock()
            .expect("OnceSceneBuffer lock poisoned")
            .pending
            .len()
    }
}

impl<T: Pod + Send + Sync + 'static> OnceGpuBufferDispatch for OnceSceneBuffer<T> {
    fn queue_handoff_bytes(&self, row: u32, data: &[u8]) {
        assert_eq!(data.len(), std::mem::size_of::<T>());
        // SAFETY: exact-size unaligned value read; `T: Pod` accepts every
        // source bit pattern admitted by the caller's registered type.
        let value = unsafe { std::ptr::read_unaligned(data.as_ptr() as *const T) };
        self.queue_handoff(row, value);
    }

    fn flush(&self, queue: &wgpu::Queue) -> SyncStats {
        OnceSceneBuffer::flush(self, queue)
    }

    fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        OnceSceneBuffer::reserve(self, queue, capacity)
    }

    fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool {
        OnceSceneBuffer::shrink_to_fit(self, queue, highest_live_row, slack_factor)
    }

    fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        OnceSceneBuffer::with_buffer(self, f)
    }

    fn buffer_snapshot(&self) -> (wgpu::Buffer, u64) {
        OnceSceneBuffer::buffer_snapshot(self)
    }

    fn epoch(&self) -> u64 {
        OnceSceneBuffer::epoch(self)
    }

    fn capacity(&self) -> u32 {
        OnceSceneBuffer::capacity(self)
    }

    fn pending_count(&self) -> usize {
        OnceSceneBuffer::pending_count(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_once_handoff_does_not_initialize_scatter_pipeline() {
        let (device, queue) = crate::gpu::test_support::test_gpu();
        let buf = OnceSceneBuffer::<u32>::new(device, "once-lazy-scatter", 4);
        assert!(
            buf.scatter.get().is_none(),
            "registration must not eagerly require a compute pipeline"
        );
        assert!(
            buf.state.lock().unwrap().scatter_scratch.is_none(),
            "registration must not eagerly allocate scatter staging buffers"
        );

        buf.queue_handoff(0, 17);
        let stats = buf.flush(&queue);
        assert_eq!((stats.ranges, stats.bytes), (1, 4));
        assert!(
            buf.scatter.get().is_none(),
            "the direct-write path must not initialize scatter"
        );
        assert!(
            buf.state.lock().unwrap().scatter_scratch.is_none(),
            "the direct-write path must not allocate scatter staging buffers"
        );
    }
}
