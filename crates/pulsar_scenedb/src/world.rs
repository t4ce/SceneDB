use crate::archetype::{Archetype, ArchetypeId, ArchetypeKey};
use crate::component::{Column, Component, ComponentId, ErasedColumn};
use crate::entity::{Entity, EntitySlot};
use crate::replication::ChangeTracker;
use ahash::AHashMap;

/// The central ECS store: owns all entities, their component data, and the
/// archetype graph.
///
/// # Entity lifecycle
///
/// 1. [`World::spawn`] allocates a slot and places the entity in the empty
///    archetype (no components).
/// 2. [`World::insert`] adds a component, migrating the entity to a new
///    archetype.
/// 3. [`World::remove`] strips a component, migrating back.
/// 4. [`World::despawn`] frees the slot and swap-removes the entity from its
///    archetype.
///
/// # Queries
///
/// Use [`World::query`] to iterate entities matching a component pattern.
/// Queries scan all archetypes, using the `u64` bitmask to skip non-matching
/// archetypes in constant time.
pub struct World {
    pub(crate) entity_slots: Vec<EntitySlot>,
    pub(crate) free_slots: Vec<u32>,
    pub(crate) archetypes: Vec<Archetype>,
    pub(crate) archetype_index: AHashMap<ArchetypeKey, ArchetypeId>,
    /// GPU-mirror wiring for `#[gpu]`-tagged component fields (see
    /// `crate::gpu::world_mirror`). `None` (the default) means `insert`
    /// behaves exactly as it does without the `gpu` feature at all — this
    /// field, and the one check against it in `insert_inner`, are the
    /// entire surface area this crate's GPU layer adds to `World` itself.
    /// Compiled out completely without the `gpu` feature (CONTRACTS C0:
    /// `--no-default-features` never depends on `wgpu`).
    #[cfg(feature = "gpu")]
    gpu_mirror: Option<crate::gpu::GpuMirrorHandle>,
}

impl World {
    /// Create an empty world with one empty archetype and no entities.
    pub fn new() -> Self {
        let empty = Archetype::new_empty(ArchetypeId::EMPTY);
        let mut archetype_index = AHashMap::default();
        archetype_index.insert(ArchetypeKey(vec![]), ArchetypeId::EMPTY);
        Self {
            entity_slots: Vec::new(),
            free_slots: Vec::new(),
            archetypes: vec![empty],
            archetype_index,
            #[cfg(feature = "gpu")]
            gpu_mirror: None,
        }
    }

    /// Attach a [`crate::gpu::GpuMirrorHandle`] so that every future
    /// `insert`/`insert_tracked` call automatically mirrors any `#[gpu]`
    /// fields of the inserted component to their registered GPU buffer, at a
    /// stable component-local row — no per-call opt-in needed once this is
    /// set. Renderer projections must resolve that row through
    /// [`Self::gpu_row`] rather than infer it from `entity.index()`.
    ///
    /// Call this once during setup, after registering every World-mirrored
    /// component with generated `T::register_gpu_columns_growable`. The
    /// fixed `register_gpu_columns` method is the CellStorage path and has no
    /// per-component World presence buffer; using such a registration through
    /// `World` fails loudly rather than claiming removal-safe semantics.
    ///
    /// Idempotent: calling this again replaces the previous handle, it does
    /// not stack. Passing a handle backed by a different, smaller-capacity
    /// `SceneGpuStore` than a previous one is the caller's responsibility to
    /// avoid — this method does no capacity reconciliation of its own.
    #[cfg(feature = "gpu")]
    pub fn attach_gpu_mirror(&mut self, mirror: crate::gpu::GpuMirrorHandle) {
        self.gpu_mirror = Some(mirror);
    }

    /// Detach the GPU mirror, if any — subsequent inserts stop writing to
    /// the GPU (existing GPU-side data is left as-is, now unmaintained).
    #[cfg(feature = "gpu")]
    pub fn detach_gpu_mirror(&mut self) -> Option<crate::gpu::GpuMirrorHandle> {
        self.gpu_mirror.take()
    }

    /// Whether a GPU mirror is currently attached (see
    /// [`Self::attach_gpu_mirror`]).
    #[cfg(feature = "gpu")]
    pub fn has_gpu_mirror(&self) -> bool {
        self.gpu_mirror.is_some()
    }

    /// The currently-attached GPU mirror, if any — e.g. to reach
    /// [`crate::gpu::GpuMirrorHandle::generations`] for binding the
    /// liveness/generation buffer into a shader. `GpuMirrorHandle` is cheap
    /// to `Clone`, so callers that already kept their own copy from before
    /// [`Self::attach_gpu_mirror`] don't need this — it exists for the case
    /// where they didn't.
    #[cfg(feature = "gpu")]
    pub fn gpu_mirror(&self) -> Option<&crate::gpu::GpuMirrorHandle> {
        self.gpu_mirror.as_ref()
    }

    /// Returns `entity`'s component-local GPU row for `T` in the currently
    /// attached mirror.
    ///
    /// The public [`Entity`] remains stable and generation-bearing; only GPU
    /// storage is component-local. Returns `None` when no mirror is attached,
    /// `T` has not been handed to it, or that component lifetime ended.
    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu_row<T: Component>(&self, entity: Entity) -> Option<u32> {
        self.gpu_mirror.as_ref()?.gpu_row::<T>(entity)
    }

    /// Type-erased counterpart to [`Self::gpu_row`].
    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu_row_for_component(&self, component_id: ComponentId, entity: Entity) -> Option<u32> {
        self.gpu_mirror
            .as_ref()?
            .gpu_row_for_component(component_id, entity)
    }

    /// Uploads every row queued since the last call — both `#[gpu(mirror =
    /// DirtyTracked)]` World-mirrored fields (the default `#[gpu]` mode) and,
    /// as of SceneDB#39, `#[gpu(mirror = Once)]` fields and the liveness/
    /// generation mirror, all of which now defer their writes here too
    /// instead of writing immediately inline with `spawn`/`insert`/`despawn`
    /// — coalesced into as few GPU writes as row adjacency allows. Call once
    /// per frame — the World-mirror analogue of the cell-mirrored path's own
    /// boundary-phase sync. `None` if no mirror is attached;
    /// `Some(SyncStats::default()-shaped)` (zero ranges/bytes) if one is
    /// attached but nothing was pending.
    #[cfg(feature = "gpu")]
    pub fn flush_gpu_mirror(&self, queue: &wgpu::Queue) -> Option<crate::gpu::SyncStats> {
        self.gpu_mirror.as_ref().map(|m| {
            let mut total = m.generations().flush(queue);
            let store = m.store().flush_gpu_mirror(queue);
            total.ranges += store.ranges;
            total.bytes += store.bytes;
            total
        })
    }

    /// Reserves capacity `n` on every registered World-mirrored GPU buffer
    /// right now, ahead of a known-size batch of upcoming inserts — moves
    /// what would otherwise be an unpredictable, mid-batch reallocation
    /// (a real GPU-to-GPU copy, potentially tens of milliseconds at
    /// AAA-relevant scale — see Helio#211's benchmark findings) off the
    /// per-insert critical path and onto this one, caller-controlled call.
    /// `None` if no mirror is attached (nothing to reserve on); `Some(Err(..))`
    /// if some registered column can't grow that far (e.g. the device's own
    /// `wgpu::Limits::max_buffer_size` — see
    /// `gpu::DynamicGpuBuffer::ensure_capacity`'s doc).
    ///
    /// Call this before a batch, not instead of sizing
    /// `register_gpu_columns_growable`'s `initial_capacity` sensibly — the
    /// two aren't redundant: `initial_capacity` is what a *fresh* `World`
    /// starts with, `reserve` is for growing an *already-running* `World`
    /// ahead of a specific future batch.
    #[cfg(feature = "gpu")]
    pub fn reserve_gpu_mirror_capacity(
        &self,
        queue: &wgpu::Queue,
        n: u32,
    ) -> Option<Result<(), crate::gpu::CapacityError>> {
        self.gpu_mirror.as_ref().map(|m| {
            m.generations().reserve(queue, n)?;
            m.store().reserve_world_mirror_capacity(queue, n)
        })
    }

    /// Reserve only component `T`'s local value and presence buffers.
    ///
    /// Unlike [`Self::reserve_gpu_mirror_capacity`], this does not pre-grow
    /// unrelated component buffers or the global entity-generation mirror.
    /// Returns `None` when no mirror is attached.
    #[cfg(feature = "gpu")]
    pub fn reserve_gpu_component_capacity<T: Component>(
        &self,
        capacity: u32,
    ) -> Option<Result<(), crate::gpu::CapacityError>> {
        self.gpu_mirror
            .as_ref()
            .map(|mirror| mirror.reserve_gpu_component_capacity::<T>(capacity))
    }

    /// Shrinks every registered World-mirrored GPU buffer to the smallest
    /// capacity that still covers `highest_live_row`, with `slack_factor`
    /// headroom (e.g. `1.5` = 50% extra room before the next growth).
    /// `highest_live_row` must come from the caller — `World` doesn't
    /// currently expose a ready-made "highest live `Entity::index()`" query
    /// on its own; a caller can compute one from `entity_slots.len()` minus
    /// its own trailing-freed-slot bookkeeping, or from whatever tracks
    /// object counts already. Call at a natural boundary (a level unload, a
    /// large despawn batch settling), not every frame — this is a real
    /// GPU-to-GPU copy per buffer that actually shrinks. No-op if no mirror
    /// is attached.
    #[cfg(feature = "gpu")]
    pub fn shrink_gpu_mirror_to_fit(
        &self,
        queue: &wgpu::Queue,
        highest_live_row: u32,
        slack_factor: f32,
    ) {
        if let Some(mirror) = &self.gpu_mirror {
            mirror
                .generations()
                .shrink_to_fit(queue, highest_live_row, slack_factor);
            mirror
                .store()
                .shrink_world_mirror_to_fit(queue, highest_live_row, slack_factor);
        }
    }

    /// Shrink only component `T`'s local value and presence buffers to the
    /// row span tracked by the attached mirror. Surviving rows remain stable;
    /// only the physical allocation and its epoch may change.
    #[cfg(feature = "gpu")]
    pub fn shrink_gpu_component_to_fit<T: Component>(&self, slack_factor: f32) -> Option<bool> {
        self.gpu_mirror
            .as_ref()
            .map(|mirror| mirror.shrink_gpu_component_to_fit::<T>(slack_factor))
    }

    /// Debug assertion: every archetype's column lengths must equal its entity
    /// count.  Panics on the first mismatch.  Compiled out in release builds
    /// (the loop body becomes a no-op).
    #[inline]
    pub fn assert_archetype_consistency(&self) {
        #[cfg(debug_assertions)]
        for arch in &self.archetypes {
            let elen = arch.entities.len();
            for (cidx, col) in arch.columns.iter().enumerate() {
                if let Some(c) = col {
                    assert_eq!(
                        c.len(),
                        elen,
                        "ArchetypeId({}) column[{}] len {} != entities.len {} (key={:?})",
                        arch.id.0,
                        cidx,
                        c.len(),
                        elen,
                        arch.key.0,
                    );
                }
            }
        }
    }

    /// Number of archetype layouts currently allocated, including the empty
    /// archetype and layouts whose last entity has been removed.
    #[inline]
    pub fn archetype_count(&self) -> usize {
        self.archetypes.len()
    }

    /// Number of archetype layouts that currently contain at least one
    /// entity. Internal archetype vectors remain sealed so external safe code
    /// cannot invalidate the storage invariants used by queries.
    #[inline]
    pub fn non_empty_archetype_count(&self) -> usize {
        self.archetypes
            .iter()
            .filter(|archetype| !archetype.is_empty())
            .count()
    }

    // â”€â”€ Entity lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Pre-allocate storage for `count` entities.  Call before a batch spawn
    /// loop to avoid repeated capacity-doubling reallocations of the slot vec
    /// and the empty archetype's entity vec.
    pub fn reserve_entities(&mut self, count: u32) {
        self.entity_slots.reserve(count as usize);
        self.archetypes[ArchetypeId::EMPTY.0 as usize]
            .entities
            .reserve(count as usize);
    }

    /// Allocate a new entity in the empty archetype.
    ///
    /// Recycles a free slot if one is available; otherwise extends the slot
    /// vec.  The returned handle includes a generation counter that allows
    /// [`is_alive`](Self::is_alive) to detect stale handles after despawn.
    pub fn spawn(&mut self) -> Entity {
        self.spawn_inner(None)
    }

    /// Like [`spawn`](Self::spawn) but also records the spawn in a
    /// [`ChangeTracker`] for replication.
    pub fn spawn_tracked(&mut self, tracker: &mut ChangeTracker) -> Entity {
        self.spawn_inner(Some(tracker))
    }

    fn spawn_inner(&mut self, tracker: Option<&mut ChangeTracker>) -> Entity {
        let (idx, gen) = if let Some(idx) = self.free_slots.pop() {
            let slot = &mut self.entity_slots[idx as usize];
            debug_assert_eq!(slot.row, crate::entity::DEAD_ROW);
            debug_assert_ne!(
                slot.generation,
                u32::MAX,
                "permanently retired entity slot entered the free list"
            );
            slot.archetype = ArchetypeId::EMPTY;
            (idx, slot.generation)
        } else {
            let idx = self.entity_slots.len() as u32;
            self.entity_slots.push(EntitySlot::empty(0));
            (idx, 0)
        };

        let entity = Entity::new(idx, gen);
        let empty = &mut self.archetypes[ArchetypeId::EMPTY.0 as usize];
        let row = empty.entities.len() as u32;
        empty.entities.push(entity);
        self.entity_slots[idx as usize].row = row;

        // GPU liveness mirror: deliberately NOT touched here (SceneDB#39).
        // A freshly spawned entity has no components yet, so at this point
        // there is no way to know whether it will ever carry a `#[gpu]`
        // field -- writing (or even queuing) a generation entry for every
        // spawn regardless would mean every entity in the World pays a
        // GPU-mirror cost the instant a mirror is attached, whether or not
        // it ever touches GPU-mirrored data. Instead, `insert_inner` queues
        // this row's generation the first time (if ever) it actually
        // receives a `#[gpu]`-bearing component -- see
        // `crate::gpu::world_mirror::GenerationMirror::note_gpu_bearing_insert`'s
        // doc for the full contract this splits across spawn/insert/despawn.

        if let Some(t) = tracker {
            t.record_spawn(entity);
        }

        entity
    }

    /// Remove an entity and all its components from the world.
    ///
    /// Returns `false` if the entity was already dead (generation mismatch or
    /// out-of-bounds index).
    ///
    /// The entity's slot is recycled: the generation is incremented and the
    /// index is pushed onto the free list.  The entity's data is
    /// swap-removed from its archetype.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.despawn_inner(entity, None)
    }

    /// Like [`despawn`](Self::despawn) but also records the despawn in a
    /// [`ChangeTracker`] for replication.
    pub fn despawn_tracked(&mut self, entity: Entity, tracker: &mut ChangeTracker) -> bool {
        self.despawn_inner(entity, Some(tracker))
    }

    fn despawn_inner(&mut self, entity: Entity, tracker: Option<&mut ChangeTracker>) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let (arch_id, row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };

        // A generation bump invalidates the whole entity identity, but GPU
        // consumers also need each component's own presence tombstone so a
        // recycled row cannot inherit component presence. Queue value-row
        // zeroing before CPU columns disappear; the generated clear callback
        // is concrete per component and does not inspect the removed value.
        #[cfg(feature = "gpu")]
        if let Some(mirror) = &self.gpu_mirror {
            let arch = &self.archetypes[arch_id.0 as usize];
            for &cid in &arch.active_cids {
                if let (Some(clear), Some(gpu_row)) = (
                    crate::gpu::world_mirror::clear_for(cid),
                    mirror.gpu_row_for_component(cid, entity),
                ) {
                    let transitioned = mirror.store().mark_component_absent(cid, gpu_row);
                    assert!(
                        transitioned.is_some(),
                        "GPU-bearing component {cid:?} was used through World without its \
                         component-presence buffer; register it with \
                         register_gpu_columns_growable (fixed register_gpu_columns is the \
                         CellStorage path and cannot provide generic removal safety)"
                    );
                    clear(mirror, gpu_row);
                    let released = mirror.release_gpu_row(cid, entity);
                    debug_assert_eq!(released, Some(gpu_row));
                }
            }
        }

        let swapped = self.archetypes[arch_id.0 as usize].remove_row(row);
        if let Some(moved) = swapped {
            self.entity_slots[moved.index() as usize].row = row as u32;
        }
        let slot = &mut self.entity_slots[entity.index() as usize];
        // u32::MAX is a permanent tombstone: normal allocation never issues
        // it as a live generation, so an ancient Entity can never become
        // valid again after counter exhaustion.
        slot.generation = slot.generation.saturating_add(1);
        slot.archetype = ArchetypeId::EMPTY;
        slot.row = crate::entity::DEAD_ROW;
        #[cfg(feature = "gpu")]
        let new_generation = slot.generation;
        if slot.generation != u32::MAX {
            self.free_slots.push(entity.index());
        }

        // GPU liveness mirror: queue the FRESHLY-BUMPED generation, not
        // `entity`'s own (now-dead) one -- a reader still holding `entity`
        // must see a mismatch against this row going forward, exactly what
        // CPU-side `is_alive` already guarantees for `entity.generation()`
        // vs. `entity_slots[idx].generation`. A genuine no-op (SceneDB#39)
        // if this row never received a `#[gpu]`-bearing component in the
        // first place -- see `GenerationMirror::note_despawn`'s doc. Queued
        // for the next `flush_gpu_mirror`, not written immediately; the
        // "reader must see a mismatch going forward" guarantee only needs to
        // hold by the next point anything actually reads GPU-mirrored state,
        // which is already gated on a flush having happened (same standing
        // assumption every other World-mirrored write in this crate rests
        // on).
        #[cfg(feature = "gpu")]
        if let Some(mirror) = &self.gpu_mirror {
            mirror
                .generations()
                .note_despawn(entity.index(), new_generation);
        }

        if let Some(t) = tracker {
            t.record_despawn(entity);
        }

        true
    }

    /// Returns `true` if `entity` is still alive.
    ///
    /// Checks that the slot exists (index in bounds) and that the stored
    /// generation matches the entity handle's generation â€” meaning the slot
    /// hasn't been recycled since the handle was created.
    #[inline]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entity_slots
            .get(entity.index() as usize)
            .map(|s| {
                s.row != crate::entity::DEAD_ROW && s.generation == entity.generation()
            })
            .unwrap_or(false)
    }

    // â”€â”€ Component helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Fast path: check whether archetype `arch_id` has a column at `cid`.
    #[inline]
    fn has_column_id(arch: &Archetype, cid: ComponentId) -> bool {
        let idx = cid.0 as usize;
        idx < arch.columns.len() && arch.columns[idx].is_some()
    }

    /// Get a mutable reference to the `ErasedColumn` at `cid` in `arch`.
    #[inline]
    fn get_erased_mut(
        arch: &mut Archetype,
        cid: ComponentId,
    ) -> Option<&mut Box<dyn ErasedColumn>> {
        arch.columns
            .get_mut(cid.0 as usize)
            .and_then(|c| c.as_mut())
    }

    /// Get a shared reference to the `ErasedColumn` at `cid` in `arch`.
    #[inline]
    fn get_erased(arch: &Archetype, cid: ComponentId) -> Option<&Box<dyn ErasedColumn>> {
        arch.columns.get(cid.0 as usize).and_then(|c| c.as_ref())
    }

    /// Ensure the columns vec is large enough for `cid`, then set it.
    #[inline]
    fn set_column(arch: &mut Archetype, cid: ComponentId, col: Box<dyn ErasedColumn>) {
        let idx = cid.0 as usize;
        for _ in arch.columns.len()..=idx {
            arch.columns.push(None);
        }
        arch.columns[idx] = Some(col);
    }

    /// Collect all CIDs that have a column in this archetype (for migration).
    fn collect_cids(arch: &Archetype) -> Vec<ComponentId> {
        arch.columns
            .iter()
            .enumerate()
            .filter(|(_, col)| col.is_some())
            .map(|(i, _)| ComponentId(i as u32))
            .collect()
    }

    /// Collect all CIDs except `skip` (for migration skip).
    fn collect_cids_skip(arch: &Archetype, skip: ComponentId) -> Vec<ComponentId> {
        arch.columns
            .iter()
            .enumerate()
            .filter(|(i, col)| col.is_some() && ComponentId(*i as u32) != skip)
            .map(|(i, _)| ComponentId(i as u32))
            .collect()
    }

    // â”€â”€ Component operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Add a component to an entity, migrating it to a new archetype if needed.
    ///
    /// If the entity already has a component of type `T`, the value is
    /// overwritten in place (no migration).  Otherwise the entity is moved to
    /// an archetype that includes `T`, preserving all existing component data.
    ///
    /// # Panics
    ///
    /// Panics if `entity` is dead.
    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) {
        self.insert_inner(entity, value, None);
    }

    /// Like [`insert`](Self::insert) but also records the change in a
    /// [`ChangeTracker`] for replication.
    pub fn insert_tracked<T: Component>(
        &mut self,
        entity: Entity,
        value: T,
        tracker: &mut ChangeTracker,
    ) {
        self.insert_inner(entity, value, Some(tracker));
    }

    fn insert_inner<T: Component>(
        &mut self,
        entity: Entity,
        value: T,
        mut tracker: Option<&mut ChangeTracker>,
    ) {
        let cid = crate::component::component_id::<T>();
        assert!(self.is_alive(entity), "insert on dead entity {entity}");

        let (old_arch_id, old_row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let is_new_insert = !Self::has_column_id(&self.archetypes[old_arch_id.0 as usize], cid);

        // GPU mirror: automatic, via a link-time dispatch registry keyed by
        // `cid` (see `crate::gpu::world_mirror`'s module docs for why this
        // is a registry lookup and not compile-time specialization — the
        // short version: `insert_inner` is itself generic, and Rust cannot
        // specialize a method call inside an unconstrained generic body on
        // the caller's substituted type). Reuses `cid`, already computed
        // above for archetype indexing — no extra `TypeId` resolution.
        // Must run before `value` is moved into a column below (both
        // branches move it), and covers BOTH the in-place-update and the
        // new-archetype-migration paths with one call, since either way
        // this is the new authoritative value for `entity`'s `T`. Passes
        // the component-presence transition through so `Once`-mode `#[gpu]`
        // fields know whether this is a new lifetime handoff. This differs
        // intentionally from `is_new_insert`: after late mirror attachment,
        // an in-place insert is still the first handoff to that mirror. A no-op
        // (one `Option::is_none()` check, then at most one `HashMap` lookup
        // that misses for any type `#[derive(SceneStore)]` never touched)
        // when no mirror is attached or `T` has no `#[gpu]` fields.
        #[cfg(feature = "gpu")]
        if let Some(mirror) = &self.gpu_mirror {
            if let Some(dispatch) = crate::gpu::world_mirror::dispatch_for(cid) {
                // For an in-place replacement the old value is still live in
                // its archetype column until after dispatch returns. Passing
                // that borrowed pointer lets the concrete generated function
                // compare only its GPU fields without cloning `T`, allocating
                // descriptor metadata, or type-erasing owned bytes. A
                // structural insertion has no prior `T` and passes `None`.
                let old_data = if is_new_insert {
                    None
                } else {
                    let old = &self.archetypes[old_arch_id.0 as usize]
                        .column::<T>()
                        .data[old_row];
                    Some(old as *const T as *const ())
                };
                // The generation gate is internally idempotent, so notify it
                // on every concrete GPU dispatch. This is necessary when a
                // mirror is attached after the CPU component already exists:
                // that insert is structurally an update, but it is the first
                // GPU lifetime in the newly attached mirror.
                mirror
                    .generations()
                    .note_gpu_bearing_insert(entity.index(), entity.generation());
                let gpu_row = mirror.acquire_gpu_row(cid, entity);
                let presence_transition = mirror.store().mark_component_present(cid, gpu_row);
                assert!(
                    presence_transition.is_some(),
                    "GPU-bearing component {cid:?} was used through World without its \
                     component-presence buffer; register it with \
                     register_gpu_columns_growable (fixed register_gpu_columns is the \
                     CellStorage path and cannot provide generic removal safety)"
                );
                let first_handoff = presence_transition.unwrap_or(is_new_insert);
                dispatch(
                    mirror,
                    gpu_row,
                    &value as *const T as *const (),
                    old_data,
                    first_handoff,
                );
            }
        }

        // In-place update: entity already has this component in this archetype.
        if !is_new_insert {
            if let Some(t) = tracker.as_deref_mut() {
                // `Component` deliberately permits non-Pod values and
                // ordinary Rust structs with padding. Reading the whole `T`
                // as bytes here would therefore observe uninitialised padding
                // (UB) and would not represent a multi-field replication
                // schema correctly anyway. Record the structural dirty event
                // only, just like the archetype-migration path below. Actual
                // field payloads must be produced through the registry's
                // per-field `Replicable` encoders (or recorded explicitly by
                // the caller) rather than guessed from `T`'s memory layout.
                t.record_component_change(entity, cid, 0, Vec::new());
            }
            let col = self.archetypes[old_arch_id.0 as usize].column_mut::<T>();
            col.data[old_row] = value;
            return;
        }

        // Build the destination archetype key and ensure it exists.
        let new_key = self.archetypes[old_arch_id.0 as usize].key.with::<T>();
        let new_arch_id = self.get_or_create_archetype(new_key);

        // Ensure Column<T> exists in the destination (may be empty).
        let new_arch = &mut self.archetypes[new_arch_id.0 as usize];
        let idx = cid.0 as usize;
        if let Some(existing) = new_arch.columns.get(idx).and_then(|c| c.as_ref()) {
            debug_assert_eq!(
                ErasedColumn::type_id(existing.as_ref()),
                std::any::TypeId::of::<T>(),
                "insert column type collision at {:?}",
                cid,
            );
        } else {
            Self::set_column(new_arch, cid, Box::new(Column::<T>::new()));
        }

        // Phase 1: push entity + migrate all existing components.
        // migrate_row pushes the entity to the destination first, then
        // transfers every column from the source, then updates all slots.
        self.migrate_row(entity, old_arch_id, old_row, new_arch_id);

        // Phase 2: push the new value.  The destination entity vec has
        // already grown by one, so this keeps all column lengths in sync.
        let new_arch = &mut self.archetypes[new_arch_id.0 as usize];
        new_arch.columns[idx]
            .as_mut()
            .unwrap()
            .as_any_mut()
            .downcast_mut::<Column<T>>()
            .unwrap()
            .data
            .push(value);

        if let Some(t) = tracker {
            // The value was moved into the column, so we can't read it anymore.
            // Record the change without field data — R2 schema encoding will handle it.
            t.record_component_change(entity, cid, 0, Vec::new());
        }
    }

    /// Remove a component from an entity, returning its value.
    ///
    /// The entity is migrated to an archetype without `T`.  All other
    /// components are preserved.
    ///
    /// Returns `None` if the entity is dead or does not have component `T`.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> Option<T> {
        self.remove_inner(entity, None)
    }

    /// Like [`remove`](Self::remove) but also records the change in a
    /// [`ChangeTracker`] for replication.
    pub fn remove_tracked<T: Component>(
        &mut self,
        entity: Entity,
        tracker: &mut ChangeTracker,
    ) -> Option<T> {
        self.remove_inner(entity, Some(tracker))
    }

    fn remove_inner<T: Component>(
        &mut self,
        entity: Entity,
        tracker: Option<&mut ChangeTracker>,
    ) -> Option<T> {
        if !self.is_alive(entity) {
            return None;
        }
        let (old_arch_id, old_row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let cid = crate::component::component_id::<T>();
        if !Self::has_column_id(&self.archetypes[old_arch_id.0 as usize], cid) {
            return None;
        }

        self.retire_gpu_component(entity, cid);

        // Pull the value out of the column.
        let removed_ptr = unsafe {
            Self::get_erased_mut(&mut self.archetypes[old_arch_id.0 as usize], cid)
                .unwrap()
                .swap_remove_erased(old_row)
        };
        // SAFETY: we know the concrete type from the generic.
        let removed_val = unsafe { *Box::from_raw(removed_ptr as *mut T) };

        // Build the destination key WITHOUT this component.
        let new_key = self.archetypes[old_arch_id.0 as usize].key.without::<T>();
        let new_arch_id = self.get_or_create_archetype(new_key);

        // Migrate everything except the removed component.
        // migrate_row_skip pushes the entity first, migrates all columns
        // except the skipped one, then updates all slots.
        self.migrate_row_skip(entity, old_arch_id, old_row, new_arch_id, cid);

        if let Some(t) = tracker {
            t.record_component_removal(entity, cid);
        }

        Some(removed_val)
    }

    /// Remove a component identified only by its runtime id.
    ///
    /// Replication needs this structural operation for an explicit
    /// component-removal delta. The erased column itself owns the concrete
    /// destructor, so non-Pod components are dropped normally and no value
    /// bytes are reinterpreted here.
    pub(crate) fn remove_component_erased(
        &mut self,
        entity: Entity,
        cid: ComponentId,
    ) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        let (old_arch_id, old_row) = {
            let slot = &self.entity_slots[entity.index() as usize];
            (slot.archetype, slot.row as usize)
        };
        if !Self::has_column_id(&self.archetypes[old_arch_id.0 as usize], cid) {
            return false;
        }

        self.retire_gpu_component(entity, cid);

        let removed_ptr = unsafe {
            Self::get_erased_mut(&mut self.archetypes[old_arch_id.0 as usize], cid)
                .expect("component presence was checked above")
                .swap_remove_erased(old_row)
        };
        unsafe {
            Self::get_erased(&self.archetypes[old_arch_id.0 as usize], cid)
                .expect("component column remains registered on its archetype")
                .drop_erased(removed_ptr);
        }

        let new_key = self.archetypes[old_arch_id.0 as usize]
            .key
            .without_id(cid);
        let new_arch_id = self.get_or_create_archetype(new_key);
        self.migrate_row_skip(entity, old_arch_id, old_row, new_arch_id, cid);
        true
    }

    /// Publish a component-presence tombstone and release its stable local
    /// GPU row before the authoritative CPU component is removed.
    #[inline]
    fn retire_gpu_component(&self, entity: Entity, cid: ComponentId) {
        #[cfg(not(feature = "gpu"))]
        let _ = (entity, cid);

        #[cfg(feature = "gpu")]
        if let Some(mirror) = &self.gpu_mirror {
            if let (Some(clear), Some(gpu_row)) = (
                crate::gpu::world_mirror::clear_for(cid),
                mirror.gpu_row_for_component(cid, entity),
            ) {
                let transitioned = mirror.store().mark_component_absent(cid, gpu_row);
                assert!(
                    transitioned.is_some(),
                    "GPU-bearing component {cid:?} was used through World without its \
                     component-presence buffer; register it with \
                     register_gpu_columns_growable (fixed register_gpu_columns is the \
                     CellStorage path and cannot provide generic removal safety)"
                );
                clear(mirror, gpu_row);
                let released = mirror.release_gpu_row(cid, entity);
                debug_assert_eq!(released, Some(gpu_row));
            }
        }
    }

    /// Returns a shared reference to component `T` on `entity`, if present.
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        if !self.is_alive(entity) {
            return None;
        }
        let s = &self.entity_slots[entity.index() as usize];
        let arch = &self.archetypes[s.archetype.0 as usize];
        let cid = crate::component::component_id::<T>();
        Self::get_erased(arch, cid).and_then(|c| {
            c.as_any()
                .downcast_ref::<Column<T>>()
                .map(|col| &col.data[s.row as usize])
        })
    }

    /// Copy-edit-replace a component through the normal insertion path.
    ///
    /// This is the mutation API for GPU-partnered components: replacing the
    /// value lets generated mirror dispatch compare the old and new fields
    /// and publish only the changed GPU rows. The edit is transactional with
    /// respect to panics because the authoritative stored value is not
    /// replaced until `edit` returns.
    ///
    /// `#[gpu(mirror = Once)]` fields retain their one-time-handoff semantics
    /// on replacement. Such fields are not suitable for authored mutation;
    /// remove and reinsert the component to begin a new presence lifetime.
    pub fn edit<T: Component + Copy, R>(
        &mut self,
        entity: Entity,
        edit: impl FnOnce(&mut T) -> R,
    ) -> Option<R> {
        let mut value = *self.get::<T>(entity)?;
        let result = edit(&mut value);
        self.insert(entity, value);
        Some(result)
    }

    /// Copy-edit-replace every selected component `T` without allocating an
    /// intermediate entity list.
    ///
    /// `edit` receives a copy of each authoritative value and returns whether
    /// that copy should be committed. Committed values go through the normal
    /// [`Self::insert`] path, so derive-generated GPU mirror dispatch observes
    /// the old/new pair and publishes only changed partner fields. Returning
    /// `false` discards the copy without touching either CPU or GPU state.
    ///
    /// The current authoritative row is not modified until the callback
    /// returns, preserving [`Self::edit`]'s transactional panic behavior. The
    /// callback cannot structurally mutate the World, so each archetype's row
    /// order remains stable for the duration of this allocation-free walk.
    ///
    /// `#[gpu(mirror = Once)]` fields retain their one-time-handoff semantics
    /// on replacement. Callers must remove and reinsert a Once component to
    /// begin a new presence lifetime.
    pub fn edit_each<T: Component + Copy>(
        &mut self,
        mut edit: impl FnMut(Entity, &mut T) -> bool,
    ) -> usize {
        let cid = crate::component::component_id::<T>();
        let mut committed = 0usize;

        for archetype_index in 0..self.archetypes.len() {
            let row_count = {
                let archetype = &self.archetypes[archetype_index];
                if Self::get_erased(archetype, cid).is_none() {
                    continue;
                }
                archetype.entities.len()
            };

            for row in 0..row_count {
                // Extract the stable identity in its own borrow window. A
                // same-component insert replaces in place and therefore cannot
                // migrate this archetype or invalidate subsequent rows.
                let entity = self.archetypes[archetype_index].entities[row];
                let mut value = *self
                    .get::<T>(entity)
                    .expect("matching archetype row lost its component during edit_each");
                if !edit(entity, &mut value) {
                    continue;
                }
                self.insert(entity, value);
                committed += 1;
            }
        }

        committed
    }

    /// Returns a mutable reference to CPU-only component `T` on `entity`, if
    /// present.
    ///
    /// # Panics
    ///
    /// When a GPU mirror is attached, this rejects component types with
    /// `#[gpu]` fields. A raw mutable reference cannot trigger generated
    /// field-level dirty dispatch; use [`Self::edit`] or [`Self::insert`]
    /// instead.
    #[inline]
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        if !self.is_alive(entity) {
            return None;
        }
        let (arch_id, row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let cid = crate::component::component_id::<T>();

        #[cfg(feature = "gpu")]
        assert!(
            self.gpu_mirror.is_none()
                || crate::gpu::world_mirror::dispatch_for(cid).is_none(),
            "get_mut cannot borrow GPU-mirrored component {:?} mutably; use World::edit or World::insert so mirror dirty dispatch runs",
            cid,
        );

        Self::get_erased_mut(&mut self.archetypes[arch_id.0 as usize], cid).and_then(|c| {
            c.as_any_mut()
                .downcast_mut::<Column<T>>()
                .map(|col| &mut col.data[row])
        })
    }

    // â”€â”€ Archetype graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Spawn `entity` at its EXACT wire index+generation, directly into the
    /// archetype identified by `key`, using `row_ops` to construct any
    /// column the archetype doesn't already have and to fill a placeholder
    /// row in every column of that archetype.
    ///
    /// Used by [`crate::replication::Delta::apply`]: replicated entity
    /// handles are shared verbatim between peers (see `Entity::bits`/
    /// `from_bits` and the replication module doc's "Endianness is a
    /// non-concern" tenet), so a replicated spawn must land at the same
    /// slot the wire value encodes, not the next locally-available one. If
    /// that slot is already live under a different occupant, the incoming
    /// spawn is authoritative and the old occupant is despawned first.
    ///
    /// Every column of the destination archetype (not just the ones newly
    /// created) is grown by one row via its
    /// [`crate::replication::RowOps::push_default`] — a real `T::default()`
    /// push, not a raw byte fill, so this is sound for ANY component type
    /// that implements `Default` (not just `Pod` ones).
    /// [`crate::replication::Delta::apply`] overwrites the real values
    /// afterward via `component_deltas`.
    ///
    /// Returns `None` (leaving the archetype registered but without the
    /// entity) if `row_ops` can't produce ops for one of `key`'s component
    /// ids, **or** if placing `entity` would grow `entity_slots` by more
    /// than `MAX_SLOT_GROWTH_PER_SPAWN` in this one call (see that local
    /// constant's doc, below — a wire-supplied index near `u32::MAX` must
    /// not be able to force a multi-gigabyte allocation).
    pub(crate) fn force_spawn_in_archetype(
        &mut self,
        entity: Entity,
        key: ArchetypeKey,
        mut row_ops: impl FnMut(ComponentId) -> Option<crate::replication::RowOps>,
    ) -> Option<Entity> {
        let idx = entity.index();
        let gen = entity.generation();

        // `u32::MAX` is the permanent-retirement generation. Accepting it as
        // a live wire entity would leave no distinct generation to publish
        // on despawn, defeating both CPU stale-handle rejection and the GPU
        // generation mirror.
        if gen == u32::MAX {
            return None;
        }

        // `idx` is wire-supplied (`Entity::index()` off a replicated,
        // attacker-controlled `Delta::spawned` entry) and the loop below
        // must grow `entity_slots`/`free_slots` to at least `idx + 1` to
        // place the entity there. Unbounded, that lets a single spawn with
        // an index near `u32::MAX` force a huge allocation before a single
        // real entity has been created — confirmed by fuzzing
        // (`delta_apply`'s `oom-*` crash artifacts): a wire index of
        // ~4.06e9 drove `entity_slots`'s `Vec<EntitySlot>` (12 bytes/elem)
        // to a real `malloc(6_442_450_944)` — a single ~6 GiB reallocation
        // (`Vec`'s doubling growth landing on 2^29 elements) — before any
        // of the earlier, smaller reallocations even had a chance to fail
        // gracefully.
        //
        // Bounding *this call's* growth (not the world's total size) is
        // the right invariant: a legitimately large, long-lived world
        // still reaches millions of entity slots over its lifetime just
        // fine, because that growth happens incrementally — one entity at
        // a time, across many separate `Delta`s — never as one call
        // demanding millions of new slots at once the way a single
        // malicious spawn does.
        const MAX_SLOT_GROWTH_PER_SPAWN: u32 = 1 << 20; // ~1M slots, ~12 MiB of EntitySlot
        let current_len = self.entity_slots.len() as u32;
        if idx > current_len && idx - current_len > MAX_SLOT_GROWTH_PER_SPAWN {
            return None;
        }

        // Resolve every type-erased constructor before mutating slots,
        // archetypes, or an existing occupant. Previously, an archetype whose
        // columns already existed could reach the post-push `row_ops(cid)?`
        // loop and return `None` after its entity vector/slot had already been
        // changed, leaving parallel column lengths inconsistent.
        let row_ops_by_component: Vec<_> = key
            .0
            .iter()
            .copied()
            .map(|cid| row_ops(cid).map(|ops| (cid, ops)))
            .collect::<Option<_>>()?;

        if let Some(slot) = self.entity_slots.get(idx as usize) {
            if gen < slot.generation || slot.generation == u32::MAX {
                // Never move a slot's generation backwards: doing so could
                // make an already-issued stale Entity live again. A MAX slot
                // is permanently retired even if an untrusted wire delta
                // targets it.
                return None;
            }
        }

        while (self.entity_slots.len() as u32) <= idx {
            let new_idx = self.entity_slots.len() as u32;
            self.entity_slots.push(EntitySlot::empty(0));
            self.free_slots.push(new_idx);
        }

        let slot = &self.entity_slots[idx as usize];
        if slot.row != crate::entity::DEAD_ROW {
            // `idx` is currently live under some other occupant — the
            // incoming delta is authoritative.
            let old_gen = slot.generation;
            self.despawn(Entity::new(idx, old_gen));
        }
        self.free_slots.retain(|&s| s != idx);
        self.entity_slots[idx as usize] = EntitySlot {
            generation: gen,
            archetype: ArchetypeId::EMPTY,
            row: 0,
        };

        let arch_id = self.get_or_create_archetype(key.clone());
        for &(cid, ops) in &row_ops_by_component {
            if !Self::has_column_id(&self.archetypes[arch_id.0 as usize], cid) {
                let col = (ops.new_column)();
                Self::set_column(&mut self.archetypes[arch_id.0 as usize], cid, col);
            }
        }

        let row = self.archetypes[arch_id.0 as usize].entities.len() as u32;
        self.archetypes[arch_id.0 as usize].entities.push(entity);
        self.entity_slots[idx as usize].archetype = arch_id;
        self.entity_slots[idx as usize].row = row;

        for &(cid, ops) in &row_ops_by_component {
            let col = Self::get_erased_mut(&mut self.archetypes[arch_id.0 as usize], cid).unwrap();
            (ops.push_default)(col.as_mut());
        }

        Some(entity)
    }

    /// Write one field's value into `entity`'s existing column for `cid` at
    /// its row, via the field's own
    /// [`crate::replication::FieldOps::decode_into`] closure — no raw
    /// pointers, no byte-width guessing; the closure downcasts to the
    /// concrete column type itself.
    ///
    /// A dead entity or a missing column is a silent no-op (`Ok(())`) —
    /// expected in ordinary operation (e.g. a stale delta arriving after a
    /// despawn, or a relevance change) — but a genuine decode failure from
    /// `decode_into` itself (malformed bytes) is propagated as `Err`.
    pub(crate) fn write_component_field(
        &mut self,
        entity: Entity,
        cid: ComponentId,
        decode_into: &(dyn Fn(&mut dyn ErasedColumn, usize, &[u8]) -> Result<(), crate::replication::ErrorCode>
              + Send
              + Sync),
        bytes: &[u8],
    ) -> Result<(), crate::replication::ErrorCode> {
        if !self.is_alive(entity) {
            return Ok(());
        }
        let (arch_id, row) = {
            let s = &self.entity_slots[entity.index() as usize];
            (s.archetype, s.row as usize)
        };
        let Some(col) = Self::get_erased_mut(&mut self.archetypes[arch_id.0 as usize], cid) else {
            return Ok(());
        };
        decode_into(col.as_mut(), row, bytes)
    }

    /// Publish the current authoritative value of one type-erased component
    /// through the World GPU mirror, if that component has generated partner
    /// dispatch and a mirror is attached.
    ///
    /// Replication reconstructs components through per-field erased decoders,
    /// not `World::insert<T>`, so it calls this once after all fields for a
    /// delta have been applied. Supplying no old value deliberately uploads
    /// every DirtyTracked partner field; Once fields upload only when the
    /// component-presence transition is new.
    pub(crate) fn publish_component_to_gpu(&self, entity: Entity, cid: ComponentId) {
        #[cfg(not(feature = "gpu"))]
        let _ = (entity, cid);

        #[cfg(feature = "gpu")]
        {
            if !self.is_alive(entity) {
                return;
            }
            let Some(mirror) = &self.gpu_mirror else {
                return;
            };
            let Some(dispatch) = crate::gpu::world_mirror::dispatch_for(cid) else {
                return;
            };
            let slot = &self.entity_slots[entity.index() as usize];
            let Some(column) = Self::get_erased(
                &self.archetypes[slot.archetype.0 as usize],
                cid,
            ) else {
                return;
            };
            // SAFETY: `slot.row` belongs to this live entity and the erased
            // column is selected by the same ComponentId used to retrieve the
            // derive-generated concrete dispatch function.
            let data = unsafe { column.get_raw(slot.row as usize) };

            mirror
                .generations()
                .note_gpu_bearing_insert(entity.index(), entity.generation());
            let gpu_row = mirror.acquire_gpu_row(cid, entity);
            let presence_transition = mirror.store().mark_component_present(cid, gpu_row);
            assert!(
                presence_transition.is_some(),
                "GPU-bearing replicated component {cid:?} has no growable presence buffer"
            );
            dispatch(
                mirror,
                gpu_row,
                data,
                None,
                presence_transition.unwrap_or(false),
            );
        }
    }

    pub(crate) fn get_or_create_archetype(&mut self, key: ArchetypeKey) -> ArchetypeId {
        if let Some(&id) = self.archetype_index.get(&key) {
            return id;
        }
        let id = ArchetypeId(self.archetypes.len() as u32);
        self.archetypes.push(Archetype::new(id, key.clone()));
        self.archetype_index.insert(key, id);
        id
    }

    // â”€â”€ Archetype migration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Move the entity and all component data from `old_arch_id`/`old_row`
    /// into `new_arch_id`.
    ///
    /// Order of operations (single cohesive window):
    /// 1. Push entity to destination `.entities` (first).
    /// 2. For each component in `active_cids` of the source archetype:
    ///    swap-remove from the source column, ensure the destination column
    ///    exists, and push into it.
    /// 3. Swap-remove the entity from the source archetype and fix the
    ///    swapped-in entity's slot row.
    /// 4. Update the migrated entity's slot.
    ///
    /// The caller is responsible for pushing any *new* component value (not
    /// present in the source archetype) *after* this returns.
    fn migrate_row(
        &mut self,
        entity: Entity,
        old_arch_id: ArchetypeId,
        old_row: usize,
        new_arch_id: ArchetypeId,
    ) {
        // Phase 1: push entity to destination first.
        let new_row = self.archetypes[new_arch_id.0 as usize].entities.len() as u32;
        self.archetypes[new_arch_id.0 as usize]
            .entities
            .push(entity);

        // Phase 2: broadcast each source column into the destination using
        // the pre-computed `active_cids` slice (no heap allocation).
        let n = self.archetypes[old_arch_id.0 as usize].active_cids.len();
        for i in 0..n {
            let cid = {
                // isolated immutable borrow â€” released before the mutable one
                let src = &self.archetypes[old_arch_id.0 as usize];
                src.active_cids[i]
            };
            let ptr = unsafe {
                Self::get_erased_mut(&mut self.archetypes[old_arch_id.0 as usize], cid)
                    .unwrap()
                    .swap_remove_erased(old_row)
            };
            if !Self::has_column_id(&self.archetypes[new_arch_id.0 as usize], cid) {
                let proto = Self::get_erased(&self.archetypes[old_arch_id.0 as usize], cid)
                    .unwrap()
                    .new_empty();
                Self::set_column(&mut self.archetypes[new_arch_id.0 as usize], cid, proto);
            }
            unsafe {
                Self::get_erased_mut(&mut self.archetypes[new_arch_id.0 as usize], cid)
                    .unwrap()
                    .push_erased(ptr);
            }
        }

        // Phase 3: remove entity from old archetype; fix swapped-in slot.
        let moved = {
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            old_arch.entities.swap_remove(old_row);
            if old_row < old_arch.entities.len() {
                Some(old_arch.entities[old_row])
            } else {
                None
            }
        };
        if let Some(m) = moved {
            self.entity_slots[m.index() as usize].row = old_row as u32;
        }

        // Phase 4: update the migrated entity's slot.
        let slot = &mut self.entity_slots[entity.index() as usize];
        slot.archetype = new_arch_id;
        slot.row = new_row;
    }

    /// Move all components EXCEPT `skip_cid` and push the entity into the
    /// destination archetype.
    ///
    /// Same ordering as [`migrate_row`]: entity first, then columns, then
    /// slot updates.
    fn migrate_row_skip(
        &mut self,
        entity: Entity,
        old_arch_id: ArchetypeId,
        old_row: usize,
        new_arch_id: ArchetypeId,
        skip_cid: ComponentId,
    ) {
        // Phase 1: push entity to destination first.
        let new_row = self.archetypes[new_arch_id.0 as usize].entities.len() as u32;
        self.archetypes[new_arch_id.0 as usize]
            .entities
            .push(entity);

        // Phase 2: migrate all columns except `skip_cid`.
        let n = self.archetypes[old_arch_id.0 as usize].active_cids.len();
        for i in 0..n {
            let cid = {
                let src = &self.archetypes[old_arch_id.0 as usize];
                src.active_cids[i]
            };
            if cid == skip_cid {
                continue;
            }
            let ptr = unsafe {
                Self::get_erased_mut(&mut self.archetypes[old_arch_id.0 as usize], cid)
                    .unwrap()
                    .swap_remove_erased(old_row)
            };
            if !Self::has_column_id(&self.archetypes[new_arch_id.0 as usize], cid) {
                let proto = Self::get_erased(&self.archetypes[old_arch_id.0 as usize], cid)
                    .unwrap()
                    .new_empty();
                Self::set_column(&mut self.archetypes[new_arch_id.0 as usize], cid, proto);
            }
            unsafe {
                Self::get_erased_mut(&mut self.archetypes[new_arch_id.0 as usize], cid)
                    .unwrap()
                    .push_erased(ptr);
            }
        }

        // Phase 3: remove entity from old archetype; fix swapped-in slot.
        let moved = {
            let old_arch = &mut self.archetypes[old_arch_id.0 as usize];
            old_arch.entities.swap_remove(old_row);
            if old_row < old_arch.entities.len() {
                Some(old_arch.entities[old_row])
            } else {
                None
            }
        };
        if let Some(m) = moved {
            self.entity_slots[m.index() as usize].row = old_row as u32;
        }

        // Phase 4: update the migrated entity's slot.
        let slot = &mut self.entity_slots[entity.index() as usize];
        slot.archetype = new_arch_id;
        slot.row = new_row;
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[test]
    fn forged_next_generation_does_not_make_a_free_slot_alive() {
        let mut world = World::new();
        let old = world.spawn();
        assert!(world.despawn(old));

        let forged = Entity::new(old.index(), old.generation() + 1);
        assert!(
            !world.is_alive(forged),
            "matching a free slot's stored generation is not occupancy"
        );
        assert!(world.get::<u32>(forged).is_none());
        assert!(!world.despawn(forged));

        let replacement = world.spawn();
        assert_eq!(replacement.index(), old.index());
        assert_eq!(replacement.generation(), forged.generation());
        assert!(world.is_alive(replacement));
    }

    #[test]
    fn exhausted_entity_slot_is_permanently_retired() {
        let mut world = World::new();
        let initial = world.spawn();
        world.entity_slots[initial.index() as usize].generation = u32::MAX - 1;
        let exhausted = Entity::new(initial.index(), u32::MAX - 1);

        assert!(world.despawn(exhausted));
        assert!(!world.is_alive(exhausted));
        assert_eq!(
            world.entity_slots[initial.index() as usize].generation,
            u32::MAX
        );
        assert_eq!(
            world.entity_slots[initial.index() as usize].row,
            crate::entity::DEAD_ROW
        );

        let next = world.spawn();
        assert_ne!(next.index(), exhausted.index());
    }

    #[test]
    fn replicated_spawn_cannot_issue_retirement_generation() {
        let mut world = World::new();
        let retired_generation = Entity::new(0, u32::MAX);
        assert!(
            world
                .force_spawn_in_archetype(
                    retired_generation,
                    ArchetypeKey::new(Vec::new()),
                    |_| None,
                )
                .is_none()
        );
        assert!(world.entity_slots.is_empty());
    }
}
