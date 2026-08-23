//! Growable, dirty-tracked GPU column for World-mirrored
//! `#[gpu(mirror = DirtyTracked)]` fields (the default `#[gpu]` mode) —
//! writes are recorded (marked dirty, CPU-side) rather than uploaded
//! immediately; [`World::flush_gpu_mirror`](crate::world::World::flush_gpu_mirror)
//! performs the actual, coalesced upload once per call.
//!
//! # Why a CPU-side shadow, not a re-read from `World`'s own storage
//!
//! The cell-mirrored path's `SceneBuffer::sync_region` coalesces dirty rows
//! by reading straight from `CellStorage`'s own per-field SoA columns — those
//! columns are already indexed by the exact row the GPU buffer uses.
//! `World`'s archetype storage has no equivalent: a component's data lives
//! in `Column<T>` at an ARCHETYPE ROW, which moves on migration/compaction
//! and has no relationship to its stable component-local GPU row. There is
//! no existing "read component `T` in GPU-row order" primitive to read from.
//!
//! So this type keeps its own CPU-side shadow — `Vec<T>`, indexed by
//! the same component-local row directly — as the coalescing scan's source
//! of truth. This costs real CPU memory (one shadow row per mirrored row,
//! same size as the GPU element), which is the honest price of
//! deferring/coalescing writes for data that isn't naturally row-indexed on
//! the CPU side the way `CellStorage` already is.
//!
//! # Scope decision: a second coalescing-scan implementation, not a shared one
//!
//! [`Self::flush`]'s run-detection loop mirrors `SceneBuffer::sync_region`'s
//! algorithm (strict adjacency — no `GAP_MERGE_THRESHOLD`-style gap bridging;
//! see that constant's own extensively-benchmarked doc for why strict
//! adjacency is the right default) rather than sharing an implementation
//! with it. Refactoring `SceneBuffer::sync_region` — already shipped,
//! already perf-validated — to serve both this type and the cell-mirrored
//! path wasn't judged worth the risk to land alongside this feature without
//! its own benchmarking pass; a reasonable follow-up, not done here.
//!
//! # Concurrency (Helio#213)
//!
//! [`Self::mark_dirty`] used to take the outer `RwLock`'s **write** lock
//! unconditionally, even when no growth was needed — serializing every
//! concurrent mark against every other one regardless of which (disjoint)
//! rows they targeted. Fixed via [`ShadowRow`]: the shadow is
//! `Vec<ShadowRow<T>>` instead of `Vec<T>`, where each element supports
//! `&self`-based get/set. The `UnsafeCell` is protected by a fixed set of
//! row-striped mutexes: same-row writes serialize, while unrelated rows only
//! contend on the rare hash collision. This is important because
//! `mark_dirty(&self, ...)` is a public safe API; soundness cannot depend on
//! callers informally promising never to race one row.
//!
//! `mark_dirty`'s common case (row already fits) still only needs the outer
//! lock's **read** side. Growth needs its write side (reallocating the shadow
//! Vec requires exclusive access), and [`Self::flush`] holds that write side
//! throughout, so neither can overlap any fast-path shadow write.
//!
//! **What this does NOT fix.** Helio#211's benchmark measured contention
//! through `GrowableSceneBuffer::write_row_growing` (the growable map, not
//! this dirty-tracked one) — a path that *already* used the same
//! read-lock-first shape this fix brings to `mark_dirty`, and it *still*
//! showed per-write cost increasing with thread count. That strongly
//! suggests a meaningful share of the observed contention is `wgpu::Queue`'s
//! own internal synchronization (a `Queue` is `Send + Sync` precisely
//! because `wgpu` serializes submissions to it internally), not solely
//! locking on SceneDB's own side — no amount of restructuring *this*
//! crate's locks removes synchronization that lives inside `wgpu` itself.
//! A genuinely lock-free (not just read-lock-first) redesign, or a
//! batch-writes-per-thread-then-one-call scheme, remain open follow-ups if
//! a future benchmark pass shows this fix + reservation (Helio#212,
//! removing *growth* from the concurrent path entirely) aren't sufficient —
//! deliberately not committed to speculatively here.
use crate::gpu::dynamic_buffer::CapacityError;
use crate::gpu::scatter_write::ScatterWritePipeline;
use crate::gpu::{DirtyMask, DynamicGpuBuffer, SyncStats};
use crate::page::Pod;
use std::cell::UnsafeCell;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// Fixed synchronization overhead per dirty-tracked buffer. This avoids a
/// mutex beside every row (which would dominate compact u32 columns), while
/// making same-row writes safe. The mask keeps the modulo branch-free.
const ROW_LOCK_SHARDS: usize = 1024;

/// How a DirtyTracked column preserves its contents when its physical GPU
/// allocation changes.
///
/// `GpuCopy` is the normal high-throughput path. `RewriteFromCpuShadow` is an
/// additive portability policy for backends that support buffer creation and
/// queue writes but do not yet expose command-encoder buffer copies. It is
/// valid here because DirtyTracked owns a complete row-indexed CPU shadow;
/// Once columns deliberately cannot select this policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DirtyTrackedReallocationPolicy {
    #[default]
    GpuCopy,
    RewriteFromCpuShadow,
}

/// A single shadow row, readable/writable through `&self` — see this
/// module's "Concurrency" doc section for the full soundness argument.
/// Deliberately not `Cell<T>`: `Cell<T>` is never `Sync` for any `T`, which
/// would make `DirtyTrackedState<T>` (and therefore `DirtyTrackedSceneBuffer<T>`)
/// non-`Sync`, breaking the `Send + Sync` bound `DirtyTrackedGpuBufferDispatch`
/// already requires (needed to store this behind `Box<dyn ...>` in
/// `SceneGpuStore`, itself shared across threads via `Arc`).
struct ShadowRow<T>(UnsafeCell<T>);

// SAFETY: every shared-path write is serialized by the owning buffer's
// row-striped mutex. Reads occur only while holding the outer RwLock's write
// side, which excludes every shared-path writer. ShadowRow is private, so no
// caller can bypass those two synchronization paths.
unsafe impl<T: Send> Sync for ShadowRow<T> {}

impl<T: Copy> ShadowRow<T> {
    fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    #[inline]
    fn get(&self) -> T {
        // SAFETY: called only under the owning state's exclusive write lock.
        unsafe { *self.0.get() }
    }

    #[inline]
    fn set(&self, value: T) {
        // SAFETY: called either under the owning state's exclusive write
        // lock or while holding this row's synchronization stripe.
        unsafe {
            *self.0.get() = value;
        }
    }
}

struct DirtyTrackedState<T: Pod> {
    buf: DynamicGpuBuffer<T>,
    shadow: Vec<ShadowRow<T>>,
    dirty: DirtyMask,
    /// Scratch buffers for [`super::scatter_write`]'s GPU-side scatter path
    /// — sized to whatever the largest flush so far needed, not to `buf`'s
    /// capacity. The allocation itself is also lazy: direct-write-only
    /// columns consume no extra GPU handles/pages.
    scatter_scratch: Option<ScatterScratch>,
}

struct ScatterScratch {
    indices: DynamicGpuBuffer<u32>,
    values: DynamicGpuBuffer<u32>,
}

/// Grows `state.shadow`/`state.dirty` (pure CPU, no GPU work) to cover
/// `new_len`, if it doesn't already — shared by [`DirtyTrackedSceneBuffer::mark_dirty`]
/// (grows lazily, one row at a time as marks arrive) and
/// [`DirtyTrackedSceneBuffer::reserve`] (grows eagerly, ahead of a known
/// batch). `DirtyMask` has no resize primitive of its own, so growing it
/// means rebuilding at the new capacity and re-marking whatever was already
/// dirty — only runs when `new_len` actually exceeds the current shadow
/// length, not on every call.
fn grow_shadow_to<T: Pod>(state: &mut DirtyTrackedState<T>, new_len: usize) {
    if new_len <= state.shadow.len() {
        return;
    }
    // SAFETY: `T: Pod`'s own safety contract (`page.rs`) guarantees
    // all-zero bytes are a valid `T`.
    state.shadow.resize_with(new_len, || {
        ShadowRow::new(unsafe { std::mem::zeroed::<T>() })
    });
    let new_dirty = DirtyMask::new(new_len as u32);
    for r in 0..state.dirty.capacity() {
        if state.dirty.is_marked(r) {
            new_dirty.mark(r);
        }
    }
    state.dirty = new_dirty;
}

/// Uploads `dirty_rows` (already sorted ascending, already coalesced by the
/// caller into however many contiguous runs it forms) via one
/// `queue.write_buffer` call per run. The cheaper strategy when there are
/// few runs — see [`DirtyTrackedSceneBuffer::SCATTER_RUN_THRESHOLD`]'s doc.
fn flush_via_direct_writes<T: Pod>(
    buf: &mut DynamicGpuBuffer<T>,
    shadow: &[ShadowRow<T>],
    dirty_rows: &[u32],
    queue: &wgpu::Queue,
) -> SyncStats {
    let mut stats = SyncStats {
        ranges: 0,
        bytes: 0,
    };
    let mut run_buf: Vec<T> = Vec::new();
    let mut upload_run = |start: u32, end: u32, stats: &mut SyncStats| {
        run_buf.clear();
        run_buf.extend(
            shadow[start as usize..end as usize]
                .iter()
                .map(ShadowRow::get),
        );
        buf.write(queue, start, &run_buf);
        stats.ranges += 1;
        stats.bytes += ((end - start) as usize * std::mem::size_of::<T>()) as u64;
    };
    let mut run_start = dirty_rows[0];
    let mut run_end = run_start + 1;
    for &row in &dirty_rows[1..] {
        if row == run_end {
            run_end = row + 1;
            continue;
        }
        upload_run(run_start, run_end, &mut stats);
        run_start = row;
        run_end = row + 1;
    }
    upload_run(run_start, run_end, &mut stats);
    stats
}

/// Uploads `dirty_rows` via [`super::scatter_write::ScatterWritePipeline`]:
/// two flat, contiguous host-side arrays (row indices + their values,
/// packed in the same order) uploaded in exactly two `queue.write_buffer`
/// calls, then one compute dispatch scatters each value to its destination
/// row on the GPU. Constant GPU-call count regardless of how many rows are
/// dirty or how scattered they are — the strategy for many runs (see
/// [`DirtyTrackedSceneBuffer::SCATTER_RUN_THRESHOLD`]'s doc).
fn flush_via_scatter<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scatter: &ScatterWritePipeline,
    buf: &mut DynamicGpuBuffer<T>,
    shadow: &[ShadowRow<T>],
    scratch: &mut ScatterScratch,
    dirty_rows: &[u32],
) -> SyncStats {
    let words_per_element = (std::mem::size_of::<T>() / 4) as u32;
    let dirty_count = dirty_rows.len() as u32;

    scratch
        .indices
        .ensure_capacity(device, queue, dirty_count)
        .expect("scatter-write scratch buffer has no max_capacity -- growth cannot fail");
    scratch
        .values
        .ensure_capacity(device, queue, dirty_count * words_per_element)
        .expect("scatter-write scratch buffer has no max_capacity -- growth cannot fail");

    // Packed in the SAME order as dirty_rows: index i's value lives at
    // words [i*words_per_element, (i+1)*words_per_element) below.
    let values: Vec<T> = dirty_rows
        .iter()
        .map(|&row| shadow[row as usize].get())
        .collect();
    scratch.indices.write(queue, 0, dirty_rows);
    // SAFETY: reinterpreting `&[T]` as `&[u32]` of `values.len() *
    // words_per_element` elements -- valid because `T: Pod` (no padding/
    // niche issues) and `DirtyTrackedSceneBuffer::new` asserts
    // `size_of::<T>() % 4 == 0`, so `words_per_element` exactly accounts
    // for every byte of every `T`.
    let values_as_u32: &[u32] = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u32,
            values.len() * words_per_element as usize,
        )
    };
    scratch.values.write(queue, 0, values_as_u32);

    scatter.dispatch(
        device,
        queue,
        scratch.indices.buffer(),
        scratch.values.buffer(),
        buf.buffer(),
        words_per_element,
        dirty_count,
    );

    SyncStats {
        ranges: 1,
        bytes: dirty_count as u64 * std::mem::size_of::<T>() as u64,
    }
}

/// Type-erased counterpart, mirroring [`super::GrowableGpuBufferDispatch`]'s
/// own reasoning (the buffer lives behind a lock, so `&wgpu::Buffer` can't be
/// returned directly — [`Self::with_buffer`] is the lock-safe replacement).
pub trait DirtyTrackedGpuBufferDispatch: Send + Sync {
    /// Records `data` as row `row`'s new value and marks it dirty — no GPU
    /// work at all (not even a device/queue borrow) unless growth is needed,
    /// in which case only the CPU-side shadow/mask grow; the GPU buffer
    /// itself grows lazily, at the next [`Self::flush`].
    fn mark_dirty_bytes(&self, row: u32, data: &[u8]);

    /// Uploads every row marked dirty since the last flush, coalesced into
    /// as few `queue.write_buffer` calls as adjacency allows. Grows the GPU
    /// buffer to match the shadow's capacity first, if needed — this is the
    /// only point this type's growth can (in principle) fail, and it can't
    /// in practice, since these buffers are never registered with a
    /// `max_capacity` ceiling (see `SceneGpuStore::register_dirty_tracked_gpu_buffer`).
    fn flush(&self, queue: &wgpu::Queue) -> SyncStats;

    /// Grows the shadow and GPU buffer to `capacity` right now. See
    /// [`DirtyTrackedSceneBuffer::reserve`].
    fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError>;

    /// Shrinks the shadow and GPU buffer to fit `highest_live_row`. See
    /// [`DirtyTrackedSceneBuffer::shrink_to_fit`].
    fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool;

    fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer));

    /// Clone the current physical handle and read its epoch under one lock,
    /// so a concurrent growth cannot produce a mismatched pair.
    fn buffer_snapshot(&self) -> (wgpu::Buffer, u64);

    /// See [`DirtyTrackedSceneBuffer::epoch`].
    fn epoch(&self) -> u64;

    fn as_any(&self) -> &dyn std::any::Any;
}

pub struct DirtyTrackedSceneBuffer<T: Pod> {
    device: Arc<wgpu::Device>,
    /// Constructed only if a flush actually crosses the scatter threshold.
    /// Small/direct-write workloads therefore need no shader-module or
    /// compute-pipeline support merely to register a mirrored component.
    /// It lives outside `state`, so initialization never recursively borrows
    /// the buffer state.
    scatter: OnceLock<ScatterWritePipeline>,
    label: String,
    reallocation_policy: DirtyTrackedReallocationPolicy,
    state: RwLock<DirtyTrackedState<T>>,
    row_locks: Box<[Mutex<()>]>,
}

impl<T: Pod + Send + Sync + 'static> DirtyTrackedSceneBuffer<T> {
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
        // gpu::scatter_write's shader treats every element as whole `u32`
        // words -- see its module doc for why this is the one requirement
        // on `T` that isn't already implied by `Pod`.
        assert_eq!(
            std::mem::size_of::<T>() % 4,
            0,
            "DirtyTrackedSceneBuffer requires size_of::<T>() to be a multiple of 4 bytes \
             (scatter-write operates on whole u32 words); got {} bytes",
            std::mem::size_of::<T>(),
        );
        let buf = DynamicGpuBuffer::new(&device, label, initial_capacity);
        // SAFETY: `T: Pod`'s own safety contract (`page.rs`) guarantees
        // all-zero bytes are a valid `T` -- the same invariant `CellStorage`
        // itself relies on for freshly-allocated, not-yet-written rows.
        let shadow = (0..initial_capacity)
            .map(|_| ShadowRow::new(unsafe { std::mem::zeroed::<T>() }))
            .collect();
        let dirty = DirtyMask::new(initial_capacity);
        Self {
            scatter: OnceLock::new(),
            label: label.to_owned(),
            reallocation_policy,
            state: RwLock::new(DirtyTrackedState {
                buf,
                shadow,
                dirty,
                scatter_scratch: None,
            }),
            row_locks: (0..ROW_LOCK_SHARDS).map(|_| Mutex::new(())).collect(),
            device,
        }
    }

    #[inline]
    fn row_lock(&self, row: u32) -> &Mutex<()> {
        // Fold high row bits into the low shard index so component-local rows
        // separated by a power-of-two capacity do not always share a lock.
        let mixed = row ^ (row >> 10) ^ (row >> 20);
        &self.row_locks[mixed as usize & (ROW_LOCK_SHARDS - 1)]
    }

    /// Records `value` as row `row`'s new value and marks it dirty.
    ///
    /// Fast path (row already within the current shadow length — the
    /// overwhelming common case once a buffer has warmed up): only the
    /// outer lock's **read** side plus one row stripe is needed. Same-row
    /// writers serialize; disjoint rows proceed concurrently unless their
    /// hashes collide. See this module's "Concurrency" section.
    ///
    /// Slow path (growth needed): escalates to the write lock, same as
    /// [`super::GrowableSceneBuffer::write_row_growing`]'s own
    /// double-checked shape — re-checks under the write lock in case
    /// another thread already grew far enough while this one was waiting
    /// for it.
    pub fn mark_dirty(&self, row: u32, value: T) {
        {
            let state = self
                .state
                .read()
                .expect("DirtyTrackedSceneBuffer lock poisoned");
            if (row as usize) < state.shadow.len() {
                let _row = self
                    .row_lock(row)
                    .lock()
                    .expect("DirtyTrackedSceneBuffer row lock poisoned");
                state.shadow[row as usize].set(value);
                state.dirty.mark(row);
                return;
            }
        }
        let mut state = self
            .state
            .write()
            .expect("DirtyTrackedSceneBuffer lock poisoned");
        if row as usize >= state.shadow.len() {
            let mut new_len = state.shadow.len().max(1);
            while new_len <= row as usize {
                new_len = new_len.saturating_mul(2);
            }
            grow_shadow_to(&mut state, new_len);
        }
        state.shadow[row as usize].set(value);
        state.dirty.mark(row);
    }

    /// Grows the CPU-side shadow (and `DirtyMask`) to cover `capacity` right
    /// now, and the underlying GPU buffer to match, ahead of any
    /// `mark_dirty`/`flush` call that would otherwise trigger either
    /// lazily. See [`crate::gpu::DynamicGpuBuffer::reserve`]'s doc for the
    /// intent — moving a known-size batch's reallocation cost off the
    /// per-insert/per-flush critical path. Growing the GPU buffer here too
    /// (not just the shadow) means a `flush()` right after a reserved batch
    /// doesn't ALSO need to grow it at that point.
    pub fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        let mut state = self
            .state
            .write()
            .expect("DirtyTrackedSceneBuffer lock poisoned");
        let rebuilt_from_shadow = if state.buf.capacity() < capacity {
            match self.reallocation_policy {
                DirtyTrackedReallocationPolicy::GpuCopy => {
                    state.buf.reserve(&self.device, queue, capacity)?;
                    false
                }
                DirtyTrackedReallocationPolicy::RewriteFromCpuShadow => state
                    .buf
                    .ensure_capacity_discarding(&self.device, capacity)?,
            }
        } else {
            false
        };
        // Grow the unbounded host shadow only after DynamicGpuBuffer has
        // checked the device/configured ceiling. Otherwise an invalid huge
        // reservation can hang or OOM in Vec growth instead of returning
        // CapacityError as this API promises.
        grow_shadow_to(&mut state, capacity as usize);
        if rebuilt_from_shadow {
            for row in 0..state.shadow.len() as u32 {
                state.dirty.mark(row);
            }
        }
        Ok(())
    }

    /// Shrinks both the CPU-side shadow and the underlying GPU buffer to
    /// the smallest size that still covers `highest_live_row` (plus
    /// `slack_factor` headroom). See
    /// [`crate::gpu::DynamicGpuBuffer::shrink_to_fit`]'s doc — same
    /// semantics, applied to both halves of this type's storage together
    /// (shrinking only the GPU buffer while leaving a larger shadow around
    /// would just waste CPU memory for no benefit).
    pub fn shrink_to_fit(
        &self,
        queue: &wgpu::Queue,
        highest_live_row: u32,
        slack_factor: f32,
    ) -> bool {
        let mut state = self
            .state
            .write()
            .expect("DirtyTrackedSceneBuffer lock poisoned");
        let target = (((highest_live_row as u64 + 1) as f64 * slack_factor.max(1.0) as f64).ceil()
            as u64)
            .min(u32::MAX as u64) as usize;
        if target >= state.shadow.len() {
            return false;
        }
        state.shadow.truncate(target);
        state.shadow.shrink_to_fit();
        let new_dirty = DirtyMask::new(target as u32);
        for r in 0..(target as u32).min(state.dirty.capacity()) {
            if state.dirty.is_marked(r) {
                new_dirty.mark(r);
            }
        }
        state.dirty = new_dirty;
        let rebuilt_from_shadow = match self.reallocation_policy {
            DirtyTrackedReallocationPolicy::GpuCopy => state
                .buf
                .shrink_to_fit(&self.device, queue, highest_live_row, slack_factor),
            DirtyTrackedReallocationPolicy::RewriteFromCpuShadow => state
                .buf
                .shrink_to_fit_discarding(&self.device, highest_live_row, slack_factor),
        };
        if rebuilt_from_shadow
            && self.reallocation_policy == DirtyTrackedReallocationPolicy::RewriteFromCpuShadow
        {
            for row in 0..state.shadow.len() as u32 {
                state.dirty.mark(row);
            }
        }
        rebuilt_from_shadow
    }

    /// Above this many contiguous runs, [`Self::flush`] switches from direct
    /// `queue.write_buffer`-per-run uploads to [`super::scatter_write`]'s
    /// GPU-side scatter pass. Below it, direct writes win (scatter's fixed
    /// cost -- two buffer uploads plus a compute dispatch -- isn't paid back
    /// by avoiding just a couple of `write_buffer` calls). Above it, scatter
    /// wins decisively, because its cost is constant in run count while
    /// direct writes' is linear -- see `gpu::scatter_write`'s module doc for
    /// the real measurement this threshold is picked from (roughly uniform
    /// scattered churn at realistic entity counts produces dirty sets that
    /// are ~all runs of length 1, i.e. thousands of runs).
    const SCATTER_RUN_THRESHOLD: u32 = 4;

    pub fn flush(&self, queue: &wgpu::Queue) -> SyncStats {
        let mut state = self
            .state
            .write()
            .expect("DirtyTrackedSceneBuffer lock poisoned");
        let shadow_len = state.shadow.len() as u32;
        if state.buf.capacity() < shadow_len {
            let rebuilt_from_shadow = match self.reallocation_policy {
                DirtyTrackedReallocationPolicy::GpuCopy => {
                    state
                        .buf
                        .ensure_capacity(&self.device, queue, shadow_len)
                        .expect(
                            "DirtyTrackedSceneBuffer never sets a max_capacity -- growth cannot fail",
                        );
                    false
                }
                DirtyTrackedReallocationPolicy::RewriteFromCpuShadow => state
                    .buf
                    .ensure_capacity_discarding(&self.device, shadow_len)
                    .expect(
                        "DirtyTrackedSceneBuffer never sets a max_capacity -- growth cannot fail",
                    ),
            };
            if rebuilt_from_shadow {
                for row in 0..shadow_len {
                    state.dirty.mark(row);
                }
            }
        }

        // Gather every dirty row once -- O(capacity/64 + dirty_count), the
        // same cost the old row-by-row coalescing scan already paid
        // (Helio#214) -- and count how many contiguous runs they'd form.
        // That count decides the strategy below; the gathered list itself
        // is what BOTH strategies actually upload from.
        let mut dirty_rows: Vec<u32> = Vec::new();
        let mut run_count: u32 = 0;
        let mut run_end: u32 = 0;
        state.dirty.for_each_marked(|row| {
            if dirty_rows.is_empty() || row != run_end {
                run_count += 1;
            }
            run_end = row + 1;
            dirty_rows.push(row);
        });
        if dirty_rows.is_empty() {
            return SyncStats {
                ranges: 0,
                bytes: 0,
            };
        }

        let stats = if run_count <= Self::SCATTER_RUN_THRESHOLD {
            let DirtyTrackedState { buf, shadow, .. } = &mut *state;
            flush_via_direct_writes(buf, shadow, &dirty_rows, queue)
        } else {
            let scatter = self
                .scatter
                .get_or_init(|| ScatterWritePipeline::new(&self.device));
            if state.scatter_scratch.is_none() {
                state.scatter_scratch = Some(ScatterScratch {
                    indices: DynamicGpuBuffer::new(
                        &self.device,
                        &format!("{}-scatter-indices", self.label),
                        1,
                    ),
                    values: DynamicGpuBuffer::new(
                        &self.device,
                        &format!("{}-scatter-values", self.label),
                        1,
                    ),
                });
            }
            let DirtyTrackedState {
                buf,
                shadow,
                scatter_scratch,
                ..
            } = &mut *state;
            flush_via_scatter(
                &self.device,
                queue,
                scatter,
                buf,
                shadow,
                scatter_scratch.as_mut().expect("scatter scratch initialized"),
                &dirty_rows,
            )
        };
        state.dirty.clear_all();
        stats
    }

    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        let state = self
            .state
            .read()
            .expect("DirtyTrackedSceneBuffer lock poisoned");
        f(state.buf.buffer());
    }

    pub fn buffer_snapshot(&self) -> (wgpu::Buffer, u64) {
        let state = self
            .state
            .read()
            .expect("DirtyTrackedSceneBuffer lock poisoned");
        (state.buf.buffer().clone(), state.buf.epoch())
    }

    /// Bump count of the underlying GPU buffer since construction — grows
    /// only at [`Self::flush`] (or [`Self::reserve`]), never at
    /// [`Self::mark_dirty`] itself, so this is a direct measure of how many
    /// times the actual `wgpu::Buffer` has been reallocated. Compare against
    /// a previously observed value to know whether a bind group built
    /// against this column needs rebuilding, same as
    /// [`super::GrowableSceneBuffer::epoch`].
    pub fn epoch(&self) -> u64 {
        self.state
            .read()
            .expect("DirtyTrackedSceneBuffer lock poisoned")
            .buf
            .epoch()
    }
}

impl<T: Pod + Send + Sync + 'static> DirtyTrackedGpuBufferDispatch for DirtyTrackedSceneBuffer<T> {
    fn mark_dirty_bytes(&self, row: u32, data: &[u8]) {
        assert_eq!(
            data.len(),
            std::mem::size_of::<T>(),
            "byte slice length must equal one element"
        );
        // SAFETY: `T: Pod`, and `data` is exactly one element's worth of
        // bytes (asserted above) -- a valid, aligned read of `T` by value.
        let value: T = unsafe { std::ptr::read_unaligned(data.as_ptr() as *const T) };
        self.mark_dirty(row, value);
    }

    fn flush(&self, queue: &wgpu::Queue) -> SyncStats {
        DirtyTrackedSceneBuffer::flush(self, queue)
    }

    fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        DirtyTrackedSceneBuffer::reserve(self, queue, capacity)
    }

    fn shrink_to_fit(&self, queue: &wgpu::Queue, highest_live_row: u32, slack_factor: f32) -> bool {
        DirtyTrackedSceneBuffer::shrink_to_fit(self, queue, highest_live_row, slack_factor)
    }

    fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        DirtyTrackedSceneBuffer::with_buffer(self, f)
    }

    fn buffer_snapshot(&self) -> (wgpu::Buffer, u64) {
        DirtyTrackedSceneBuffer::buffer_snapshot(self)
    }

    fn epoch(&self) -> u64 {
        DirtyTrackedSceneBuffer::epoch(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        crate::gpu::test_support::test_gpu()
    }

    fn readback(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buf: &wgpu::Buffer,
        bytes: u64,
    ) -> Vec<u8> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().expect("mapped range").to_vec();
        staging.unmap();
        data
    }

    #[test]
    fn marking_dirty_does_not_touch_the_gpu_until_flush() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> =
            DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", 8);
        assert!(
            buf.scatter.get().is_none(),
            "registration must not eagerly require a compute pipeline"
        );
        assert!(
            buf.state.read().unwrap().scatter_scratch.is_none(),
            "registration must not eagerly allocate scatter staging buffers"
        );

        buf.mark_dirty(0, 111);
        buf.mark_dirty(1, 222);
        buf.mark_dirty(5, 555);

        // Before flush: the GPU buffer must still read as all-zero -- the
        // marks are pure CPU-side bookkeeping so far.
        let mut before = Vec::new();
        buf.with_buffer(&mut |b| before = readback(&device, &queue, b, 8 * 4));
        assert!(
            before.iter().all(|&b| b == 0),
            "no GPU write must have happened before flush"
        );

        let stats = buf.flush(&queue);
        assert!(
            buf.scatter.get().is_none(),
            "the direct-write path must not initialize scatter"
        );
        assert!(
            buf.state.read().unwrap().scatter_scratch.is_none(),
            "the direct-write path must not allocate scatter staging buffers"
        );
        // Two runs: [0,1] (adjacent, one coalesced write) and [5,5] --
        // strict adjacency (no gap bridging), matching SceneBuffer::sync_region's
        // GAP_MERGE_THRESHOLD == 0 default.
        assert_eq!(
            stats.ranges, 2,
            "row 0-1 coalesce into one run, row 5 is a separate run (gap at 2,3,4)"
        );

        let mut after = Vec::new();
        buf.with_buffer(&mut |b| after = readback(&device, &queue, b, 8 * 4));
        let at = |row: usize| u32::from_ne_bytes(after[row * 4..row * 4 + 4].try_into().unwrap());
        assert_eq!(at(0), 111);
        assert_eq!(at(1), 222);
        assert_eq!(at(2), 0, "never marked -- must stay zero");
        assert_eq!(at(5), 555);

        // A second flush with nothing newly dirty must be a true no-op.
        let stats2 = buf.flush(&queue);
        assert_eq!(stats2.ranges, 0);
        assert_eq!(stats2.bytes, 0);
    }

    #[test]
    fn mark_dirty_past_capacity_grows_the_shadow_and_the_flush_still_lands_correctly() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> =
            DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", 2);

        buf.mark_dirty(0, 1);
        buf.mark_dirty(50, 999); // far past initial capacity of 2

        let stats = buf.flush(&queue);
        assert_eq!(stats.ranges, 2);

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 51 * 4));
        assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 1);
        assert_eq!(u32::from_ne_bytes(bytes[200..204].try_into().unwrap()), 999);
    }

    #[test]
    fn reserve_grows_both_shadow_and_gpu_buffer_so_flush_needs_no_further_growth() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> =
            DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", 2);

        buf.reserve(&queue, 500).expect("reserve");
        for row in 0..500u32 {
            buf.mark_dirty(row, row * 2);
        }
        let stats = buf.flush(&queue);
        assert_eq!(
            stats.ranges, 1,
            "500 contiguous dirty rows must coalesce into one range"
        );

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 500 * 4));
        assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 0);
        assert_eq!(
            u32::from_ne_bytes(bytes[996..1000].try_into().unwrap()),
            498
        );
    }

    #[test]
    fn shrink_to_fit_reclaims_both_shadow_and_gpu_buffer() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> =
            DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", 2);
        buf.mark_dirty(999, 42);
        buf.flush(&queue);

        let shrank = buf.shrink_to_fit(&queue, 999, 1.0);
        assert!(shrank);

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, 1000 * 4));
        assert_eq!(
            bytes.len(),
            4000,
            "buffer must have actually shrunk to exactly 1000 rows"
        );
        assert_eq!(
            u32::from_ne_bytes(bytes[999 * 4..999 * 4 + 4].try_into().unwrap()),
            42
        );

        // Further marks/flushes must still work correctly post-shrink.
        buf.mark_dirty(500, 777);
        buf.flush(&queue);
        let mut bytes2 = Vec::new();
        buf.with_buffer(&mut |b| bytes2 = readback(&device, &queue, b, 1000 * 4));
        assert_eq!(
            u32::from_ne_bytes(bytes2[500 * 4..500 * 4 + 4].try_into().unwrap()),
            777
        );
    }

    #[test]
    fn concurrent_marks_to_disjoint_rows_are_all_correctly_recorded() {
        // The whole point of #213's fix: many threads marking DISJOINT rows
        // concurrently, none lost, none corrupted, via mark_dirty's
        // read-lock-first fast path (ShadowRow's &self get/set) -- not
        // serialized through a write lock for every single mark.
        let (device, queue) = test_device();
        let buf: Arc<DirtyTrackedSceneBuffer<u32>> = Arc::new(DirtyTrackedSceneBuffer::new(
            Arc::clone(&device),
            "test",
            4096,
        ));
        // Pre-reserve so every thread's marks land in the fast (read-lock)
        // path -- this test is specifically about disjoint-row concurrency,
        // not the (already write-lock-serialized, and already tested
        // elsewhere) growth path.
        buf.reserve(&queue, 4096).expect("reserve");

        const THREADS: u32 = 16;
        const PER_THREAD: u32 = 200;
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let buf = Arc::clone(&buf);
                scope.spawn(move || {
                    for i in 0..PER_THREAD {
                        let row = t * PER_THREAD + i; // disjoint across threads by construction
                        buf.mark_dirty(row, row * 7);
                    }
                });
            }
        });

        let stats = buf.flush(&queue);
        assert_eq!(
            stats.ranges, 1,
            "every row in [0, THREADS*PER_THREAD) was marked -- one contiguous run"
        );

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| {
            bytes = readback(&device, &queue, b, (THREADS * PER_THREAD) as u64 * 4)
        });
        for row in 0..(THREADS * PER_THREAD) {
            let got = u32::from_ne_bytes(
                bytes[(row * 4) as usize..(row * 4 + 4) as usize]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(got, row * 7, "row {row} must have survived concurrent marking from whichever thread owned it, uncorrupted by any other thread's marks");
        }
    }

    #[test]
    fn concurrent_marks_to_the_same_row_never_publish_a_torn_value() {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
        struct WideRow {
            words: [u32; 8],
        }
        // SAFETY: repr(C) around one u32 array has no padding, every bit
        // pattern is valid, and WideRow is Copy with no drop glue.
        unsafe impl crate::page::Pod for WideRow {}

        let (device, queue) = test_device();
        let buf = Arc::new(DirtyTrackedSceneBuffer::new(
            Arc::clone(&device),
            "same-row-race-test",
            1,
        ));
        let start = Arc::new(std::sync::Barrier::new(17));
        std::thread::scope(|scope| {
            for writer in 1..=16u32 {
                let buf = Arc::clone(&buf);
                let start = Arc::clone(&start);
                scope.spawn(move || {
                    start.wait();
                    for _ in 0..2_000 {
                        buf.mark_dirty(0, WideRow { words: [writer; 8] });
                    }
                });
            }
            start.wait();
        });

        buf.flush(&queue);
        let mut bytes = Vec::new();
        buf.with_buffer(&mut |buffer| {
            bytes = readback(&device, &queue, buffer, std::mem::size_of::<WideRow>() as u64)
        });
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
            .collect();
        assert!(
            words.iter().all(|word| *word == words[0]),
            "same-row writers published a torn row: {words:?}",
        );
        assert!((1..=16).contains(&words[0]));
    }

    /// SceneDB#39: scattered rows (well above `SCATTER_RUN_THRESHOLD`, no
    /// two adjacent) must take the GPU-side scatter path and land every
    /// value at exactly the right row, with every untouched row still
    /// reading zero — the real-device correctness proof for
    /// `gpu::scatter_write`, not just the run-count/threshold logic.
    #[test]
    fn scattered_dirty_rows_use_the_scatter_path_and_land_at_the_right_rows() {
        let (device, queue) = test_device();
        let capacity = 1000u32;
        let buf: DirtyTrackedSceneBuffer<u32> =
            DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", capacity);

        // 40 rows, no two adjacent, spread across the whole capacity --
        // 40 runs, comfortably above SCATTER_RUN_THRESHOLD (4).
        let dirty_rows: Vec<u32> = (0..40u32).map(|i| i * 23 + 1).collect();
        for &row in &dirty_rows {
            buf.mark_dirty(row, row * 1000 + 7);
        }

        let stats = buf.flush(&queue);
        assert_eq!(
            stats.ranges, 1,
            "scatter path reports one GPU operation regardless of how many rows were scattered"
        );
        assert_eq!(stats.bytes, dirty_rows.len() as u64 * 4);
        assert!(
            buf.scatter.get().is_some(),
            "the many-run path must initialize scatter on first use"
        );
        assert!(
            buf.state.read().unwrap().scatter_scratch.is_some(),
            "the many-run path must allocate staging buffers on first use"
        );

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, capacity as u64 * 4));
        let at = |row: u32| {
            u32::from_ne_bytes(
                bytes[(row * 4) as usize..(row * 4 + 4) as usize]
                    .try_into()
                    .unwrap(),
            )
        };
        let dirty_set: std::collections::HashSet<u32> = dirty_rows.iter().copied().collect();
        for row in 0..capacity {
            if dirty_set.contains(&row) {
                assert_eq!(
                    at(row),
                    row * 1000 + 7,
                    "scattered row {row} must land at exactly its own offset, not a neighbor's"
                );
            } else {
                assert_eq!(at(row), 0, "row {row} was never marked -- must stay zero, not leak a neighboring scattered write");
            }
        }
    }

    /// Same as above but for a wide, multi-word element type
    /// (`[f32; 16]`, 64 bytes = 16 `u32` words) — the scatter shader's
    /// per-element copy loop (`words_per_element` iterations) is untested
    /// by every other test in this file, which all use plain `u32`
    /// (`words_per_element == 1`, which can't catch an off-by-one in that
    /// loop).
    #[test]
    fn scatter_path_copies_every_word_of_a_wide_multi_word_element() {
        let (device, queue) = test_device();
        let capacity = 200u32;
        let buf: DirtyTrackedSceneBuffer<[f32; 16]> =
            DirtyTrackedSceneBuffer::new(Arc::clone(&device), "test", capacity);

        let dirty_rows: Vec<u32> = (0..30u32).map(|i| i * 5 + 2).collect();
        let value_for =
            |row: u32| -> [f32; 16] { core::array::from_fn(|w| row as f32 * 100.0 + w as f32) };
        for &row in &dirty_rows {
            buf.mark_dirty(row, value_for(row));
        }
        let stats = buf.flush(&queue);
        assert_eq!(
            stats.ranges, 1,
            "scatter path (40 runs's worth of isolation exceeds the threshold)"
        );

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |b| bytes = readback(&device, &queue, b, capacity as u64 * 64));
        let dirty_set: std::collections::HashSet<u32> = dirty_rows.iter().copied().collect();
        for row in 0..capacity {
            let row_bytes = &bytes[(row as usize) * 64..(row as usize) * 64 + 64];
            let got: [f32; 16] = core::array::from_fn(|w| {
                f32::from_ne_bytes(row_bytes[w * 4..w * 4 + 4].try_into().unwrap())
            });
            let expected = if dirty_set.contains(&row) {
                value_for(row)
            } else {
                [0.0; 16]
            };
            assert_eq!(
                got, expected,
                "row {row}: every one of the 16 words must match, not just the first"
            );
        }
    }

    #[test]
    fn cpu_shadow_reallocation_rewrites_every_surviving_row() {
        let (device, queue) = test_device();
        let buf: DirtyTrackedSceneBuffer<u32> =
            DirtyTrackedSceneBuffer::new_with_reallocation_policy(
                Arc::clone(&device),
                "cpu-shadow-growth",
                2,
                DirtyTrackedReallocationPolicy::RewriteFromCpuShadow,
            );

        buf.mark_dirty(0, 11);
        buf.mark_dirty(1, 22);
        buf.flush(&queue);

        // Row 5 grows the shadow and then the GPU allocation from 2 to 8.
        // The old allocation is deliberately discarded, so rows 0 and 1 can
        // survive only if the policy really rebuilds them from the shadow.
        buf.mark_dirty(5, 66);
        let stats = buf.flush(&queue);
        assert_eq!(buf.epoch(), 1);
        assert_eq!(stats.ranges, 1);
        assert_eq!(stats.bytes, 8 * 4);

        let mut bytes = Vec::new();
        buf.with_buffer(&mut |buffer| bytes = readback(&device, &queue, buffer, 8 * 4));
        let rows: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
            .collect();
        assert_eq!(rows, [11, 22, 0, 0, 0, 66, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "multiple of 4 bytes")]
    fn element_size_not_a_multiple_of_4_bytes_panics_at_construction() {
        #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
        #[repr(transparent)]
        struct ThreeBytes([u8; 3]);
        unsafe impl Pod for ThreeBytes {}

        let (device, _queue) = test_device();
        let _buf: DirtyTrackedSceneBuffer<ThreeBytes> =
            DirtyTrackedSceneBuffer::new(device, "test", 4);
    }
}
