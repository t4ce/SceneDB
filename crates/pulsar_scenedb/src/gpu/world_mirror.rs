//! Bridges [`crate::world::World`]'s entity-indexed archetype storage to
//! [`SceneGpuStore`]'s component-local GPU-mirrored columns.
//!
//! `World` is `CellStorage`/`Handle`-free by design (it's the crate's
//! archetype ECS, not the paged storage layer `#[gpu]` mirroring was
//! originally built against — see `GpuColumnSet::write_gpu`'s
//! `Handle`-taking signature). This module lets a component's `#[gpu]`
//! fields be mirrored to their registered GPU buffer anyway. Each owning
//! component type gets its own dense row allocation domain: an entity keeps
//! a stable component-local row for that component-presence lifetime, and a
//! released row is recycled by a later entity with the same component. This
//! prevents a rare component on a high-index entity from sizing its GPU
//! buffer to the global entity high-water mark.
//!
//! The mechanism is entirely additive and opt-in:
//!
//! - Nothing here runs unless a [`GpuMirrorHandle`] has been attached to a
//!   `World` via [`crate::world::World::attach_gpu_mirror`]. Until then,
//!   `World::insert` behaves exactly as it always has.
//! - Once attached, every `World::insert`/`insert_tracked` call looks up
//!   `T`'s `ComponentId` in a link-time-populated dispatch registry (see
//!   "Dispatch mechanism" below) and, if `T` has `#[gpu]` fields, writes
//!   them to their component-local row. Callers that build projection/index
//!   buffers resolve that row through [`GpuMirrorHandle::gpu_row`]; it must
//!   not be inferred from [`crate::entity::Entity::index`].
//!
//! This keeps [`crate::world`] itself free of any *public* GPU-specific
//! API surface — the new state lives behind a `#[cfg(feature = "gpu")]`
//! field and one `#[cfg(feature = "gpu")]` block in `insert_inner`, so a
//! `--no-default-features` build of this crate is byte-for-byte the
//! `World` that existed before this module — C0's actual guarantee (zero
//! GPU deps without the feature) holds exactly as before.
//!
//! # Dispatch mechanism (and why it isn't compile-time specialization)
//!
//! An earlier version of this module tried to resolve "does `T` have
//! `#[gpu]` fields" at compile time via the "autoref specialization" trick
//! (an inherent method competing with a blanket trait method, exploiting
//! Rust's inherent-beats-trait method-resolution priority). **That does not
//! work here, and the reason is worth recording precisely** so it doesn't
//! get re-attempted: `World::insert_inner<T: Component>` is itself an
//! unconstrained generic function. Rust resolves method calls inside a
//! generic function body once, using only `T`'s *declared* bounds
//! (`Component`) — never per-monomorphization — so a specialization
//! decision written inside that body can't observe whether the *substituted*
//! `T` additionally implements `GpuColumnSet`; only code written where `T`
//! is *already concrete* (e.g. inside a macro-generated, non-generic
//! function) can. Confirmed empirically two ways before landing this
//! version: (1) a minimal standalone repro of the autoref trick called from
//! inside a generic wrapper function always picked the fallback arm,
//! regardless of the concrete type substituted, while the exact same trait
//! setup called directly on concrete types (no generic wrapper) picked the
//! right arm every time; (2) `tests/world_gpu_mirror.rs`'s real-device
//! readback caught the resulting silent no-op directly (an all-zero buffer)
//! before this fix landed.
//!
//! The working mechanism is a link-time registry instead: `#[derive(SceneStore)]`
//! emits, for any type with at least one `#[gpu]` field, a **non-generic**
//! dispatch function (concrete `T` baked in at macro-expansion time) plus a
//! [`GpuMirrorRegistration`] submitted via `inventory` (the same link-time
//! registration mechanism `SubsystemRegistry`/`DynMethodRegistry` already
//! use elsewhere in this crate/`pulsar_reflection`). [`World::insert_inner`]
//! looks the registration up by the `ComponentId` it already computes for
//! archetype indexing (no extra `TypeId` resolution) — a single `HashMap`
//! lookup, paid only when a mirror is attached, same as `component_id::<T>()`
//! itself already costs a thread-local scan on every insert regardless.
//! Not literally free, but the closest achievable without nightly
//! `#[feature(specialization)]`, and correct — which the compile-time
//! attempt, despite compiling cleanly with no errors or warnings pointing
//! at the problem, was not.
use crate::component::{component_id, Component, ComponentId};
use crate::entity::Entity;
use crate::gpu::{
    CapacityError, DirtyTrackedSceneBuffer, GpuColumnDesc, GpuColumnSet, MirrorMode, SceneGpuStore,
    SyncStats,
};
use ahash::AHashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

/// Row-indexed liveness/generation buffer, mirroring `World::entity_slots`'s
/// `generation` field on the GPU, keyed by `Entity::index()`. Component value
/// and presence buffers use a separate component-local row; projection data
/// that needs generation validation must retain the entity index alongside
/// that local row. See the "Liveness" section of the
/// README for the read-side contract this exists to support: a GPU consumer
/// holding a captured `(row, generation)` pair compares `generation` against
/// this buffer's value at `row` before trusting any other World-mirrored
/// buffer's contents at that row -- the same staleness check
/// `World::is_alive` already performs on the CPU side, made available to
/// shaders.
///
/// Built on [`DirtyTrackedSceneBuffer<u32>`] — the same shadow+dirty-mask
/// mechanism `#[gpu]` (`DirtyTracked`) fields use — rather than the
/// CellStorage-oriented `GenerationBuffer` type ([`super::GenerationBuffer`])
/// or a hand-rolled pending-write queue. See "Why `DirtyTrackedSceneBuffer`,
/// not a lighter structure" below for why this is the right call even though
/// it costs one CPU-side `u32` shadow per row.
///
/// # Deferred, gated writes (SceneDB#39)
///
/// Writes here used to be immediate (`queue.write_buffer`, once per spawn
/// AND once per despawn — i.e. at least two synchronous GPU calls per churn
/// cycle) and unconditional: `World::spawn`/`despawn` wrote a generation
/// entry for every entity regardless of whether it ever carried a `#[gpu]`
/// field, since a freshly spawned entity's eventual components aren't known
/// yet at spawn time. Confirmed by reading the call sites while
/// investigating SceneDB#39 — not merely suspected. Fixed two ways,
/// together:
///
/// - **Deferred**: [`Self::note_gpu_bearing_insert`]/[`Self::note_despawn`]
///   only mark a row dirty ([`DirtyTrackedSceneBuffer::mark_dirty`], no GPU
///   work, read-lock-first fast path); the actual upload happens in
///   [`Self::flush`], coalesced the same way `#[gpu]` field flushes already
///   are. A row despawned and respawned within the same unflushed frame
///   collapses to one write of the final generation, not two.
/// - **Gated**: an entity that never receives a `#[gpu]`-bearing component
///   insert now costs *zero* GPU-mirror work, at spawn or at despawn —
///   [`GpuMirroredRows`] tracks, per row, whether it has ever actually
///   carried GPU-mirrored data, and both `note_*` methods no-op unless that
///   flag says there is something on the GPU that actually needs
///   invalidating. This is what makes "non-GPU entities are never affected"
///   true by construction rather than by convention: an entity with no
///   `#[gpu]` fields anywhere in its component set is now indistinguishable,
///   cost-wise, from spawning/despawning it with no [`GpuMirrorHandle`]
///   attached to the `World` at all.
///
/// # Why generation keeps a shadow
///
/// Generation is a mutable `u32` validity column: slot lifetimes can update
/// any row, and its 4-byte-per-row shadow enables lock-light coalescing. This
/// is distinct from component fields declared `Once`, which use
/// [`super::OnceSceneBuffer`]'s transient handoff queue and retain no
/// capacity-sized CPU value shadow after flush.
pub struct GenerationMirror {
    buf: DirtyTrackedSceneBuffer<u32>,
    gpu_mirrored_rows: GpuMirroredRows,
}

impl GenerationMirror {
    fn new(device: Arc<wgpu::Device>) -> Self {
        // Small initial capacity, unbounded growth -- matches every other
        // World-mirrored buffer's recommended (register_gpu_columns_growable)
        // configuration; see that method's doc for why World-mirrored
        // buffers specifically should never set a max_capacity ceiling.
        Self {
            buf: DirtyTrackedSceneBuffer::new(device, "scenedb-world-mirror-generations", 64),
            gpu_mirrored_rows: GpuMirroredRows::new(),
        }
    }

    /// Called from `World::insert_inner` the first time `row` receives a
    /// component with `#[gpu]` fields (i.e. exactly once per entity, not
    /// once per `#[gpu]`-bearing component it happens to carry, and never
    /// for an entity that never gets one at all). Marks `generation` (the
    /// value already assigned at spawn, just not yet communicated to the
    /// GPU) dirty for the next [`Self::flush`]. A no-op if this row was
    /// already marked — a later insert of a second `#[gpu]`-bearing
    /// component type onto the same entity doesn't change its generation,
    /// so there is nothing new to mark.
    pub(crate) fn note_gpu_bearing_insert(&self, row: u32, generation: u32) {
        if self.gpu_mirrored_rows.mark_first_time(row) {
            self.buf.mark_dirty(row, generation);
        }
    }

    /// Called from `World::despawn_inner`. Marks the freshly-bumped
    /// `new_generation` dirty for the next [`Self::flush`] — but only if
    /// this row ever actually received a `#[gpu]`-bearing insert
    /// ([`Self::note_gpu_bearing_insert`]); otherwise there is nothing on
    /// the GPU at this row that needs invalidating, and this is a complete
    /// no-op (one `GpuMirroredRows` read-lock check, nothing else). Clears
    /// the row's flag either way, so a future entity that reuses this slot
    /// starts fresh and must earn its own GPU-mirror cost by actually
    /// carrying a `#[gpu]` field, exactly like a brand-new entity would.
    pub(crate) fn note_despawn(&self, row: u32, new_generation: u32) {
        if self.gpu_mirrored_rows.clear(row) {
            self.buf.mark_dirty(row, new_generation);
        }
    }

    /// Uploads every row marked dirty since the last flush, coalesced into
    /// contiguous runs. Called from `World::flush_gpu_mirror`, alongside
    /// (not instead of) [`super::SceneGpuStore::flush_gpu_mirror`] — call
    /// both once per frame, same as before this type deferred its writes.
    pub(crate) fn flush(&self, queue: &wgpu::Queue) -> SyncStats {
        self.buf.flush(queue)
    }

    /// Pre-grows both the CPU transition state and the GPU generation
    /// buffer. This belongs in the same explicit reservation boundary as
    /// every component-value and presence buffer: otherwise a caller can
    /// reserve all value columns and still take a surprise generation-buffer
    /// reallocation on the next flush.
    pub(crate) fn reserve(&self, queue: &wgpu::Queue, capacity: u32) -> Result<(), CapacityError> {
        // Validate/grow the bounded GPU allocation first. Doing the CPU
        // `Vec<AtomicBool>` reserve first lets an impossible request such as
        // `u32::MAX` attempt a multi-gigabyte host allocation before the
        // device limit can return the promised catchable CapacityError.
        self.buf.reserve(queue, capacity)?;
        self.gpu_mirrored_rows.reserve(capacity);
        Ok(())
    }

    pub(crate) fn shrink_to_fit(
        &self,
        queue: &wgpu::Queue,
        highest_live_row: u32,
        slack_factor: f32,
    ) -> bool {
        self.gpu_mirrored_rows
            .shrink_to_fit(highest_live_row, slack_factor);
        self.buf
            .shrink_to_fit(queue, highest_live_row, slack_factor)
    }

    pub fn with_buffer(&self, f: &mut dyn FnMut(&wgpu::Buffer)) {
        self.buf.with_buffer(f);
    }

    /// Returns the physical handle and allocation epoch under one lock.
    /// Bind-group caches should use this instead of observing the handle and
    /// epoch independently across a possible concurrent grow.
    pub fn buffer_snapshot(&self) -> (wgpu::Buffer, u64) {
        self.buf.buffer_snapshot()
    }

    pub fn epoch(&self) -> u64 {
        self.buf.epoch()
    }
}

/// Per-row "has this entity ever received a `#[gpu]`-bearing component
/// insert" flags, backing [`GenerationMirror`]'s gating (see its doc).
/// `RwLock<Vec<AtomicBool>>`, not a plain `Mutex<Vec<bool>>`: the common
/// case (row already within the backing `Vec`, which is true for almost
/// every call once a `World` has warmed up) only needs the **read** side of
/// the lock plus one atomic swap -- multiple threads marking/clearing
/// *disjoint* rows proceed concurrently, the same read-lock-first shape
/// [`DirtyTrackedSceneBuffer::mark_dirty`] uses for its own shadow. Only
/// growing the backing `Vec` (a row past the current end) needs the write
/// side.
struct GpuMirroredRows {
    rows: RwLock<Vec<AtomicBool>>,
}

impl GpuMirroredRows {
    fn new() -> Self {
        Self {
            rows: RwLock::new(Vec::new()),
        }
    }

    /// Returns `true` the first time `row` is marked; returns `false` on
    /// every call after that for the same row, until [`Self::clear`] resets
    /// it.
    fn mark_first_time(&self, row: u32) -> bool {
        let idx = row as usize;
        {
            let rows = self.rows.read().expect("GpuMirroredRows lock poisoned");
            if idx < rows.len() {
                return !rows[idx].swap(true, Ordering::Relaxed);
            }
        }
        let mut rows = self.rows.write().expect("GpuMirroredRows lock poisoned");
        if idx >= rows.len() {
            rows.resize_with(idx + 1, || AtomicBool::new(false));
        }
        !rows[idx].swap(true, Ordering::Relaxed)
    }

    /// Clears `row`'s flag and returns what it was before clearing (i.e.
    /// whether the caller should treat this as "there was GPU-mirrored data
    /// here that needs invalidating"). A row past the end of the backing
    /// `Vec` (never marked) returns `false` without growing anything --
    /// read-lock only, never escalates.
    fn clear(&self, row: u32) -> bool {
        let idx = row as usize;
        let rows = self.rows.read().expect("GpuMirroredRows lock poisoned");
        if idx >= rows.len() {
            return false;
        }
        rows[idx].swap(false, Ordering::Relaxed)
    }

    fn reserve(&self, capacity: u32) {
        let mut rows = self.rows.write().expect("GpuMirroredRows lock poisoned");
        if rows.len() < capacity as usize {
            rows.resize_with(capacity as usize, || AtomicBool::new(false));
        }
    }

    fn shrink_to_fit(&self, highest_live_row: u32, slack_factor: f32) {
        let target = (((highest_live_row as u64 + 1) as f64 * slack_factor.max(1.0) as f64).ceil()
            as u64)
            .min(u32::MAX as u64) as usize;
        let mut rows = self.rows.write().expect("GpuMirroredRows lock poisoned");
        if target < rows.len() {
            rows.truncate(target);
            rows.shrink_to_fit();
        }
    }
}

/// Component-local stable row allocation for every GPU-bearing component.
///
/// Rows are dense with respect to a component type's own peak residency, not
/// the World's global entity index. Removal adds the row to a free list and
/// never relocates surviving components; that stability is important for
/// renderer-owned projection/index buffers. The exact generation-bearing
/// [`Entity`] is the map key, so a stale handle cannot resolve a row later
/// reused by a new entity generation.
#[derive(Default)]
struct ComponentGpuRows {
    // ComponentId is dense, so the outer lookup is an indexed Vec rather
    // than another hash on every World mutation. Entity is intentionally a
    // hash map within each component: a sparse per-component Vec indexed by
    // Entity.index would recreate the high-water memory problem on the CPU.
    components: RwLock<Vec<Option<ComponentRows>>>,
}

#[derive(Default)]
struct ComponentRows {
    by_entity: AHashMap<Entity, u32>,
    free_rows: Vec<u32>,
    /// Component-local occupancy. This compact byte flag avoids another
    /// entity-sized reverse table. Trimming an absent tail keeps `row_span`
    /// exact without scanning the entity map during every renderer query.
    occupied_rows: Vec<bool>,
}

impl ComponentGpuRows {
    fn acquire(&self, component_id: ComponentId, entity: Entity) -> u32 {
        let mut components = self
            .components
            .write()
            .expect("ComponentGpuRows lock poisoned");
        let component_index = component_id.0 as usize;
        if components.len() <= component_index {
            components.resize_with(component_index + 1, || None);
        }
        let rows = components[component_index].get_or_insert_with(ComponentRows::default);
        if let Some(&row) = rows.by_entity.get(&entity) {
            return row;
        }

        let row = loop {
            match rows.free_rows.pop() {
                Some(row)
                    if (row as usize) < rows.occupied_rows.len()
                        && !rows.occupied_rows[row as usize] =>
                {
                    rows.occupied_rows[row as usize] = true;
                    break row;
                }
                Some(_) => continue, // stale entry from a trimmed absent tail
                None => {
                    let row = u32::try_from(rows.occupied_rows.len())
                        .expect("component-local GPU row space exhausted");
                    rows.occupied_rows.push(true);
                    break row;
                }
            }
        };
        let previous = rows.by_entity.insert(entity, row);
        debug_assert!(previous.is_none());
        row
    }

    fn get(&self, component_id: ComponentId, entity: Entity) -> Option<u32> {
        self.components
            .read()
            .expect("ComponentGpuRows lock poisoned")
            .get(component_id.0 as usize)
            .and_then(Option::as_ref)
            .and_then(|rows| rows.by_entity.get(&entity).copied())
    }

    fn release(&self, component_id: ComponentId, entity: Entity) -> Option<u32> {
        let mut components = self
            .components
            .write()
            .expect("ComponentGpuRows lock poisoned");
        let rows = components
            .get_mut(component_id.0 as usize)
            .and_then(Option::as_mut)?;
        let row = rows.by_entity.remove(&entity)?;
        debug_assert!(rows.occupied_rows[row as usize]);
        rows.occupied_rows[row as usize] = false;
        if rows.by_entity.is_empty() {
            // A completely empty component population can restart at row 0
            // and release allocator-side high-water memory. GPU tombstones
            // have already been queued by World before this is called.
            *rows = ComponentRows::default();
        } else {
            rows.free_rows.push(row);
            while rows.occupied_rows.last() == Some(&false) {
                rows.occupied_rows.pop();
            }
        }
        Some(row)
    }

    fn live_count(&self, component_id: ComponentId) -> u32 {
        self.components
            .read()
            .expect("ComponentGpuRows lock poisoned")
            .get(component_id.0 as usize)
            .and_then(Option::as_ref)
            .map_or(0, |rows| rows.by_entity.len() as u32)
    }

    fn row_span(&self, component_id: ComponentId) -> u32 {
        self.components
            .read()
            .expect("ComponentGpuRows lock poisoned")
            .get(component_id.0 as usize)
            .and_then(Option::as_ref)
            .map_or(0, |rows| rows.occupied_rows.len() as u32)
    }
}

/// The resources `World`'s automatic GPU mirroring needs: the store every
/// `#[gpu]` field's buffer lives in, the queue to write through, the global
/// liveness/generation mirror (see [`GenerationMirror`]), and the shared
/// component-local row map.
///
/// Attach via [`crate::world::World::attach_gpu_mirror`]. Cheap to clone
/// (all state is `Arc`-backed).
#[derive(Clone)]
pub struct GpuMirrorHandle {
    store: Arc<SceneGpuStore>,
    queue: Arc<wgpu::Queue>,
    generations: Arc<GenerationMirror>,
    component_rows: Arc<ComponentGpuRows>,
}

impl GpuMirrorHandle {
    pub fn new(store: Arc<SceneGpuStore>, queue: Arc<wgpu::Queue>) -> Self {
        let generations = Arc::new(GenerationMirror::new(store.device_arc()));
        Self {
            store,
            queue,
            generations,
            component_rows: Arc::new(ComponentGpuRows::default()),
        }
    }

    #[inline]
    pub fn store(&self) -> &SceneGpuStore {
        &self.store
    }

    #[inline]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    #[inline]
    pub fn generations(&self) -> &GenerationMirror {
        &self.generations
    }

    /// Returns `entity`'s stable GPU row for component `T` in its current
    /// component-presence lifetime.
    ///
    /// This is the row renderer projections must upload. It is deliberately
    /// unrelated to `entity.index()`, which remains the row of the separate
    /// global generation/liveness buffer. Returns `None` when the entity has
    /// not handed `T` to this mirror or after `T` was removed/despawned.
    #[inline]
    pub fn gpu_row<T: Component>(&self, entity: Entity) -> Option<u32> {
        self.gpu_row_for_component(component_id::<T>(), entity)
    }

    /// Type-erased counterpart to [`Self::gpu_row`].
    #[inline]
    pub fn gpu_row_for_component(
        &self,
        component_id: ComponentId,
        entity: Entity,
    ) -> Option<u32> {
        self.component_rows.get(component_id, entity)
    }

    /// Number of currently mirrored entities carrying `T`.
    #[inline]
    pub fn gpu_live_count<T: Component>(&self) -> u32 {
        self.gpu_live_count_for_component(component_id::<T>())
    }

    /// Type-erased counterpart to [`Self::gpu_live_count`].
    #[inline]
    pub fn gpu_live_count_for_component(&self, component_id: ComponentId) -> u32 {
        self.component_rows.live_count(component_id)
    }

    /// Current addressable row span for `T`. This can exceed
    /// [`Self::gpu_live_count`] while stable rows contain holes, but is bounded
    /// by `T`'s peak concurrent mirrored population rather than by the global
    /// entity high-water mark.
    #[inline]
    pub fn gpu_row_span<T: Component>(&self) -> u32 {
        self.gpu_row_span_for_component(component_id::<T>())
    }

    /// Type-erased counterpart to [`Self::gpu_row_span`].
    #[inline]
    pub fn gpu_row_span_for_component(&self, component_id: ComponentId) -> u32 {
        self.component_rows.row_span(component_id)
    }

    /// Pre-grow only `T`'s component-local value and presence buffers.
    ///
    /// The global generation mirror is intentionally not affected. Call the
    /// existing World-wide reservation API separately when pre-growing the
    /// `Entity::index()` generation domain is also required.
    pub fn reserve_gpu_component_capacity<T: Component>(
        &self,
        capacity: u32,
    ) -> Result<(), CapacityError> {
        self.store.reserve_world_component_capacity(
            &self.queue,
            component_id::<T>(),
            capacity,
        )
    }

    /// Shrink `T`'s value and presence buffers to its current local row span.
    /// Surviving rows never move; this only changes physical capacity/epoch.
    pub fn shrink_gpu_component_to_fit<T: Component>(&self, slack_factor: f32) -> bool {
        let component_id = component_id::<T>();
        let row_span = self.component_rows.row_span(component_id);
        // wgpu storage allocations retain a one-row floor when the component
        // population is empty; DynamicGpuBuffer expresses its target as an
        // inclusive highest row, so span 0 and span 1 both use row 0 here.
        let highest_live_row = row_span.saturating_sub(1);
        self.store.shrink_world_component_to_fit(
            &self.queue,
            component_id,
            highest_live_row,
            slack_factor,
        )
    }

    #[inline]
    pub(crate) fn acquire_gpu_row(&self, component_id: ComponentId, entity: Entity) -> u32 {
        self.component_rows.acquire(component_id, entity)
    }

    #[inline]
    pub(crate) fn release_gpu_row(
        &self,
        component_id: ComponentId,
        entity: Entity,
    ) -> Option<u32> {
        self.component_rows.release(component_id, entity)
    }
}

/// Writes every `#[gpu]` field of `data` into its registered GPU buffer at
/// `row`, honoring each field's declared [`crate::gpu::MirrorMode`]
/// (`GpuColumnDesc::mode`):
///
/// - **`Once`**: queued only on an absent→present component transition (and
///   for the explicit removal tombstone), not on ordinary in-place updates.
/// - **`DirtyTracked`** (the default): marked dirty
///   ([`SceneGpuStore::mark_gpu_row_dirty`]) instead of written immediately.
///   [`crate::world::World::flush_gpu_mirror`] performs the actual,
///   coalesced upload — call it once per frame, analogous to the
///   cell-mirrored path's own boundary-phase sync.
///
/// Each field's `ComponentId` is looked up in whichever data-buffer
/// registration maps it was actually registered through (fixed
/// [`SceneGpuStore::write_row_bytes`], growable
/// [`SceneGpuStore::write_row_bytes_growing`], or dirty-tracked
/// [`SceneGpuStore::mark_gpu_row_dirty`], or Once
/// [`SceneGpuStore::queue_gpu_row_once`]) — a given field lives in exactly
/// one, so this is at most a few cheap map lookups, not redundant work. A
/// field whose buffer was never registered through any of them is silently
/// skipped, not an error — legitimate during bring-up.
///
/// Works for *any* `T: GpuColumnSet` by walking `T::gpu_columns()`. It is the
/// reflective/direct helper for hand-written implementations.
/// `#[derive(SceneStore)]` deliberately does not call it from World mutation
/// dispatch: generated insert and clear code addresses every field directly,
/// avoiding this helper's descriptor `Vec` allocation per mutation.
///
/// # Panics
///
/// Only if a field was registered growable with an explicit `max_capacity`
/// ceiling (via `register_growable_gpu_buffer`, not through the derive's
/// generated `register_gpu_columns_growable`, which never sets one) and
/// `row` exceeds it. This is deliberate, not an oversight: `World::insert`
/// has no `Result` to propagate a capacity failure through, so a caller who
/// opts into a hard ceiling on a World-mirrored column is opting into this
/// panic as the price of that ceiling — documented at
/// [`SceneGpuStore::register_growable_gpu_buffer`].
pub fn write_gpu_columns_at_row<T: GpuColumnSet>(
    store: &SceneGpuStore,
    queue: &wgpu::Queue,
    row: u32,
    data: &T,
    first_handoff: bool,
) {
    for col in T::gpu_columns() {
        let size = col.field_token.desc().size as usize;
        // SAFETY: this is exactly the unsafe contract of `GpuColumnSet`:
        // `field_offset`/`size` identify a fully initialized, padding-free
        // Pod field within `T`, and the descriptor's tokens match that field.
        // The whole component deliberately need not be Pod; CPU-only fields
        // may contain arbitrary rich Rust values.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts((data as *const T as *const u8).add(col.field_offset), size)
        };
        let id = col.field_token.id();
        write_gpu_column_bytes_at_row(store, queue, row, id, col.mode, bytes, first_handoff);
    }
}

/// Type-erased single-column counterpart to [`write_gpu_columns_at_row`].
///
/// `first_handoff` is a component-presence transition, not merely an ECS
/// structural insert flag. That distinction makes `Once` correct after
/// remove/re-insert and after attaching/registering a mirror later in an
/// entity's lifetime.
pub fn write_gpu_column_bytes_at_row(
    store: &SceneGpuStore,
    queue: &wgpu::Queue,
    row: u32,
    id: ComponentId,
    mode: MirrorMode,
    bytes: &[u8],
    first_handoff: bool,
) {
    if mode == MirrorMode::Once && !first_handoff {
        return;
    }

    let deferred = match mode {
        MirrorMode::DirtyTracked => store.mark_gpu_row_dirty(id, row, bytes),
        MirrorMode::Once => store.queue_gpu_row_once(id, row, bytes),
    };
    if deferred || store.write_row_bytes(id, queue, bytes, row) {
        return;
    }
    match store.write_row_bytes_growing(id, queue, bytes, row) {
        None => {} // not registered through any path -- bring-up, not an error
        Some(Ok(())) => {}
        Some(Err(cap_err)) => panic!(
            "World-mirrored GPU column (ComponentId {id:?}) hit its configured max_capacity \
             ({}) at row {row} (requested {}) -- see SceneGpuStore::register_growable_gpu_buffer's \
             doc: World-mirrored columns should be registered with max_capacity: None precisely \
             to make this unreachable",
            cap_err.max, cap_err.requested,
        ),
    }
}

/// Queues an all-zero value tombstone for every GPU partner of `T` at
/// `row`. The owning component's presence column is the semantic absence
/// marker; zeroing value storage is defense-in-depth so accidental consumers
/// do not observe a previous component lifetime's payload.
pub fn clear_gpu_columns_at_row<T: GpuColumnSet>(
    store: &SceneGpuStore,
    queue: &wgpu::Queue,
    row: u32,
) {
    // Never construct an all-zero `T`: GpuColumnSet only promises that its
    // described GPU fields are Pod. A component may also contain String,
    // Vec, NonZero*, references, or other CPU-only values for which a zeroed
    // whole component would be immediate UB. Zero the partnered rows by
    // descriptor instead. Removal is cold enough that one reusable scratch
    // allocation is preferable to expanding the unsafe surface.
    let columns = T::gpu_columns();
    let max_size = columns
        .iter()
        .map(|column| column.field_token.desc().size as usize)
        .max()
        .unwrap_or(0);
    let zero = vec![0_u8; max_size];
    for column in columns {
        let size = column.field_token.desc().size as usize;
        write_gpu_column_bytes_at_row(
            store,
            queue,
            row,
            column.field_token.id(),
            column.mode,
            &zero[..size],
            true,
        );
    }
}

// ── Link-time dispatch registry ─────────────────────────────────────────

/// One `#[derive(SceneStore)]` type's entry in the world-mirror dispatch
/// table: `component_id` identifies the type (a plain `fn` pointer, not the
/// already-resolved `ComponentId`, since resolving it requires the global
/// `component_id::<T>()` registry lock and must happen lazily, not at
/// `inventory::submit!`'s const-eval time); `dispatch` is a **non-generic**
/// function — `T` is already concrete at the point the derive macro
/// generates it — that downcasts the type-erased component pointers back to
/// `&T`, compares generated GPU fields directly, and dispatches only changed
/// physical rows.
///
/// Not constructed by hand — `#[derive(SceneStore)]` emits one of these
/// (via `inventory::submit!`) for every type with at least one `#[gpu]`
/// field. Types with none don't submit a registration at all, so they
/// never appear in [`registry_map`] and cost nothing beyond the one
/// `HashMap` miss `World::insert` already pays when a mirror is attached.
/// `data` and `old_data`: as documented on
/// [`GpuMirrorRegistration::dispatch`]. `old_data` is present only for an
/// in-place replacement and stays live for the duration of the call. The
/// final `bool` is the owning component's absent→present transition in this
/// mirror, so a `Once` field hands off after late attachment or
/// remove/re-insert without re-uploading on ordinary in-place updates.
pub type DispatchFn = fn(&GpuMirrorHandle, u32, *const (), Option<*const ()>, bool);
pub type ClearFn = fn(&GpuMirrorHandle, u32);
pub type DescriptorsFn = fn() -> Vec<GpuColumnDesc>;

pub struct GpuMirrorRegistration {
    pub component_id: fn() -> ComponentId,
    /// Physical World-buffer descriptors for this component. Non-packed
    /// components return their per-field descriptors; packed components
    /// return one descriptor for the packed row.
    pub descriptors: DescriptorsFn,
    /// `data` must point to a live, correctly-aligned `T` for the
    /// registration's own (macro-generated, concrete) `T` — upheld by
    /// `World::insert_inner`, the only caller, which passes `&value as *const
    /// T as *const ()` for the exact `T` this registration's `component_id`
    /// resolved from. When `old_data` is `Some`, it obeys the same type and
    /// alignment contract and points into the existing archetype column. It
    /// remains borrowed and unmoved until this function returns; new
    /// structural inserts pass `None`.
    pub dispatch: DispatchFn,
    /// Queues value tombstones for every GPU partner owned by this component.
    /// Component presence is updated by `World` before this callback runs.
    pub clear: ClearFn,
}

pulsar_reflection::inventory::collect!(GpuMirrorRegistration);

#[derive(Clone, Copy)]
struct RegistrationFns {
    dispatch: DispatchFn,
    clear: ClearFn,
    descriptors: DescriptorsFn,
}

fn registry_map() -> &'static HashMap<ComponentId, RegistrationFns> {
    static MAP: OnceLock<HashMap<ComponentId, RegistrationFns>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut map = HashMap::new();
        for registration in pulsar_reflection::inventory::iter::<GpuMirrorRegistration>() {
            let id = (registration.component_id)();
            let previous = map.insert(
                id,
                RegistrationFns {
                    dispatch: registration.dispatch,
                    clear: registration.clear,
                    descriptors: registration.descriptors,
                },
            );
            assert!(
                previous.is_none(),
                "more than one GpuMirrorRegistration was submitted for component {:?}",
                id,
            );
        }
        map
    })
}

/// Looks up `id`'s dispatch function, if `#[derive(SceneStore)]` generated
/// one for it (i.e. the type has at least one `#[gpu]` field). `id` is
/// expected to already be in hand — [`crate::world::World::insert_inner`]
/// computes it via `component_id::<T>()` for archetype indexing regardless
/// of GPU mirroring, so this adds exactly one `HashMap` lookup on top, not
/// a second `TypeId` resolution.
#[inline]
pub(crate) fn dispatch_for(id: ComponentId) -> Option<DispatchFn> {
    registry_map().get(&id).map(|entry| entry.dispatch)
}

/// Looks up the concrete tombstone writer emitted for a GPU-bearing
/// component. Removal and despawn use this independently of insertion.
#[inline]
pub(crate) fn clear_for(id: ComponentId) -> Option<ClearFn> {
    registry_map().get(&id).map(|entry| entry.clear)
}

/// Return the link-time registered physical World-buffer descriptors for a
/// component type. This reflection path does not require a `World` instance
/// or a GPU device and is generated for every `SceneStore` type with at least
/// one `#[gpu]` field. For growable World registration, the same `id` is also
/// the identity passed to
/// [`SceneGpuStore::component_presence_buffer_snapshot_for_id`]; presence is
/// deliberately owner metadata, not another value-column descriptor.
pub fn gpu_column_descs_for_component(id: ComponentId) -> Option<Vec<GpuColumnDesc>> {
    registry_map().get(&id).map(|entry| (entry.descriptors)())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuColumnDesc, MirrorMode};
    use crate::token::TypeToken;

    /// A minimal hand-rolled `GpuColumnSet` type (mirrors the shape
    /// `#[derive(SceneStore)]` would generate for one `#[gpu]` field),
    /// registered exactly the way the derive's generated code would.
    #[repr(transparent)]
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    struct TestField(u32);
    unsafe impl crate::page::Pod for TestField {}

    struct TestComponent {
        value: TestField,
        // Proves the generic clear path never zero-constructs the whole
        // component. This is intentionally non-Copy and not zero-valid.
        #[allow(dead_code)]
        cpu_only: String,
    }
    // SAFETY: the sole descriptor names `value` at its exact offset and uses
    // the padding-free `TestField` wrapper as both field and value token.
    unsafe impl GpuColumnSet for TestComponent {
        fn gpu_columns() -> Vec<GpuColumnDesc> {
            vec![GpuColumnDesc {
                field_token: TypeToken::of::<TestField>(),
                value_token: TypeToken::of::<TestField>(),
                field_offset: std::mem::offset_of!(TestComponent, value),
                mode: MirrorMode::DirtyTracked,
                buffer_name: "value",
                buffer_key: None,
            }]
        }
        fn write_gpu(
            _store: &SceneGpuStore,
            _id: crate::gpu::CellId,
            _cell: &mut crate::cell::CellStorage,
            _handle: crate::handle::Handle,
            _data: &Self,
            _phase: &impl crate::gpu::SimulateWitness,
        ) {
        }
    }

    fn test_dispatch(
        mirror: &GpuMirrorHandle,
        row: u32,
        data: *const (),
        _old_data: Option<*const ()>,
        is_new_insert: bool,
    ) {
        let data = unsafe { &*(data as *const TestComponent) };
        write_gpu_columns_at_row(mirror.store(), mirror.queue(), row, data, is_new_insert);
    }

    fn test_clear(mirror: &GpuMirrorHandle, row: u32) {
        clear_gpu_columns_at_row::<TestComponent>(mirror.store(), mirror.queue(), row);
    }

    fn test_descriptors() -> Vec<GpuColumnDesc> {
        TestComponent::gpu_columns()
    }

    pulsar_reflection::inventory::submit! {
        GpuMirrorRegistration {
            component_id: crate::component::component_id::<TestComponent>,
            descriptors: test_descriptors,
            dispatch: test_dispatch,
            clear: test_clear,
        }
    }

    #[test]
    fn a_type_with_a_submitted_registration_is_found_by_its_component_id() {
        let id = crate::component::component_id::<TestComponent>();
        assert!(
            dispatch_for(id).is_some(),
            "TestComponent submitted a GpuMirrorRegistration in this same module — must be found"
        );
        assert_eq!(
            gpu_column_descs_for_component(id),
            Some(TestComponent::gpu_columns()),
        );
    }

    #[test]
    fn a_type_with_no_registration_is_not_found() {
        struct NeverRegistered;
        let id = crate::component::component_id::<NeverRegistered>();
        assert!(dispatch_for(id).is_none());
    }

    #[test]
    fn component_rows_are_local_stable_recycled_and_generation_exact() {
        struct Object;
        struct Light;

        let object = crate::component::component_id::<Object>();
        let light = crate::component::component_id::<Light>();
        let rows = ComponentGpuRows::default();
        let high_a = Entity::from_bits(40_000);
        let high_b = Entity::from_bits(40_001);

        assert_eq!(rows.acquire(object, high_a), 0);
        assert_eq!(rows.acquire(object, high_b), 1);
        assert_eq!(rows.acquire(object, high_a), 0, "updates keep their row");
        assert_eq!(rows.acquire(light, high_a), 0, "each component starts at row 0");
        assert_eq!(rows.live_count(object), 2);
        assert_eq!(rows.row_span(object), 2);

        assert_eq!(rows.release(object, high_a), Some(0));
        assert_eq!(rows.get(object, high_a), None);
        assert_eq!(rows.get(object, high_b), Some(1), "survivors never move");

        // Same entity slot, new generation: the stale handle cannot resolve
        // the row handed to the new component lifetime.
        let recycled_entity = Entity::from_bits((1_u64 << 32) | high_a.index() as u64);
        assert_eq!(rows.acquire(object, recycled_entity), 0);
        assert_eq!(rows.get(object, high_a), None);
        assert_eq!(rows.get(object, recycled_entity), Some(0));

        assert_eq!(rows.release(object, high_b), Some(1));
        assert_eq!(rows.row_span(object), 1, "an absent tail is trimmed in O(1)");
        let high_c = Entity::from_bits(40_002);
        assert_eq!(
            rows.acquire(object, high_c),
            1,
            "allocation skips stale free-list entries after tail trimming",
        );
        assert_eq!(rows.release(object, high_c), Some(1));
        assert_eq!(rows.release(object, recycled_entity), Some(0));
        assert_eq!(rows.live_count(object), 0);
        assert_eq!(rows.row_span(object), 0, "an empty population resets its span");
    }
}
