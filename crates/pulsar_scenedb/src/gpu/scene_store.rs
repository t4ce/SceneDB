//! The SceneDB-owned multi-cell device-side store (M2b-α §2, design Rev 2):
//! region-partitioned scene SSBOs shared across every registered CELL. Each
//! cell owns a disjoint `[row_base, row_base+capacity)` slice of the
//! transform and slot-mirror buffers and a disjoint
//! `[slot_base, slot_base+capacity+headroom)` slice of the generation buffer
//! (`RegionPool`, §7). Constructed on the engine-level device context (C0).
//!
//! Mirrored columns must be written via `SceneGpuStore::write_transform` and
//! compacted via `SceneGpuStore::compact_all`; raw column access bypasses
//! dirty tracking.

use super::{
    CapacityError, ComponentPresenceBuffer, DirtyMask, DirtyTrackedGpuBufferDispatch,
    DirtyTrackedReallocationPolicy, DirtyTrackedSceneBuffer, EngineGpuContext, GenerationBuffer,
    GpuBufferDispatch, GrowableGpuBufferDispatch, GrowableSceneBuffer, OnceGpuBufferDispatch,
    OnceSceneBuffer, RegionError, RegionPool, SceneBuffer, SimulateWitness, SubmissionTracker,
    SyncStats,
};
use crate::cell::{CellStorage, PendingRetire};
use crate::component::{component_id, ComponentId};
use crate::handle::Handle;
use crate::page::Pod;
use crate::spatial::InstanceInfo;
use crate::token::HasTypeToken;
use crate::gpu::world_mirror::{GpuMirrorRegistration, RegistrationFns};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// One size class of region (design Rev 2 §2/§7): every cell registered
/// under this class gets a fixed-size `capacity`-row region, and at most
/// `max_resident_cells` such regions ever coexist.
#[derive(Debug, Clone, Copy)]
pub struct RegionClassConfig {
    pub capacity: u32,
    pub max_resident_cells: u32,
}

/// Fixed store capacities (SSBOs never reallocate — exceeding them is a hard
/// error at the call site, §8), expressed as size classes rather than one
/// flat cap (M2a's `GpuStoreConfig`).
#[derive(Debug, Clone)]
pub struct SceneGpuConfig {
    pub classes: Vec<RegionClassConfig>,
    /// Extra slots reserved per region beyond `capacity`, absorbing
    /// tombstoned (retired-but-not-yet-recycled) slots without stealing a
    /// neighbor's region (§4.1).
    pub tombstone_headroom: u32,
    /// Per-cell metadata SSBO entries (α: allocated, no writer).
    pub max_cells_metadata: u32,
}

impl SceneGpuConfig {
    /// The default tombstone headroom used across M2b-α fixtures.
    pub fn default_headroom() -> u32 {
        64
    }
}

/// How a GPU-mirrored field is synced to the GPU buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MirrorMode {
    /// Delta-synced every frame via dirty tracking.  Default for `#[gpu]`.
    DirtyTracked,
    /// One handoff per World component-presence lifetime. Ordinary updates
    /// do not re-upload; removal queues a tombstone and a later re-insertion
    /// starts a fresh handoff. The transient CPU queue is discarded at flush.
    Once,
}

/// Describes one GPU-mirrored field in a component type.
/// Collected by `GpuColumnSet::gpu_columns()` and used by the derive macro's
/// generated `write_gpu()` and by `SceneGpuStore::sync_all()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuColumnDesc {
    /// Type token of the macro-generated field wrapper. This is the CPU
    /// [`CellStorage`] column identity and the id accepted by the public
    /// `*_for_id` GPU-buffer APIs. Named partners may resolve several such
    /// field ids to one canonical physical buffer.
    pub field_token: crate::token::TypeToken,
    /// Type token of the field value before the derive macro wraps it.
    ///
    /// A wrapper is intentionally unique to `(component, field)`, so its
    /// token cannot prove that two explicitly named partners have the same
    /// row type. `value_token` supplies that proof. Named reuse requires this
    /// token, its layout, and [`Self::mode`] to match exactly.
    pub value_token: crate::token::TypeToken,
    /// Byte offset of the field within the struct (via `offset_of!`).
    pub field_offset: usize,
    /// Mirror mode: DirtyTracked (per-frame delta) or Once (bulk upload).
    pub mode: MirrorMode,
    /// Logical buffer name for GPU binding.
    pub buffer_name: &'static str,
    /// Explicit stable partner key from `#[gpu(buffer = "...")]`.
    ///
    /// `None` deliberately means "private to this field". In particular,
    /// ordinary fields named `value`, `position`, and so on must not alias
    /// merely because their display names happen to match. Only `Some(key)`
    /// participates in cross-field physical-buffer reuse. Compatible reuse
    /// across component types is restricted to fixed CellStorage, whose row
    /// regions are disjoint. Growable World registration enforces one owning
    /// component per canonical key because each owner has an independent
    /// component-local row allocator and therefore cannot safely arbitrate a
    /// shared physical row.
    pub buffer_key: Option<&'static str>,
}

/// Cold-path registry that joins CPU field-column ids to GPU buffer ids.
///
/// Component ids are dense, so aliases and descriptors use dense vectors:
/// resolving an id on a World write is one bounds check and one load rather
/// than a second hash-table probe. The only hash map is keyed by an explicit
/// logical name and is touched during registration/editor lookup, not during
/// per-row mutation.
#[derive(Default)]
struct GpuPartnerRegistry {
    aliases: Vec<Option<ComponentId>>,
    descriptors: Vec<Option<GpuColumnDesc>>,
    residencies: Vec<Option<GpuBufferResidency>>,
    /// Owning World component for each canonical physical buffer. A World
    /// row has no arbitration between two component types, so sharing one
    /// named destination across owners would otherwise be last-write-wins.
    /// Fixed CellStorage columns do not use this table: their row regions
    /// are disjoint and compatible named reuse remains supported there.
    world_owners: Vec<Option<ComponentId>>,
    named: HashMap<&'static str, ComponentId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuBufferResidency {
    FixedCell,
    GrowableWorld,
    DirtyTrackedWorld,
    OnceWorld,
}

impl GpuPartnerRegistry {
    #[inline]
    fn resolve(&self, id: ComponentId) -> ComponentId {
        self.aliases
            .get(id.0 as usize)
            .and_then(|entry| *entry)
            .unwrap_or(id)
    }

    #[inline]
    fn descriptor(&self, id: ComponentId) -> Option<GpuColumnDesc> {
        self.descriptors.get(id.0 as usize).and_then(|entry| *entry)
    }

    fn id_for_key(&self, key: &str) -> Option<ComponentId> {
        self.named.get(key).copied()
    }

    fn register(&mut self, desc: GpuColumnDesc) {
        let field_id = desc.field_token.id();
        assert_eq!(
            desc.field_token.desc(),
            desc.value_token.desc(),
            "GPU partner `{}` has different field-wrapper and value layouts: {:?} vs {:?}",
            desc.buffer_key.unwrap_or(desc.buffer_name),
            desc.field_token.desc(),
            desc.value_token.desc(),
        );
        if let Some(key) = desc.buffer_key {
            assert!(
                !key.is_empty(),
                "#[gpu(buffer = ...)] key must not be empty"
            );
        }

        let idx = field_id.0 as usize;
        if self.descriptors.len() <= idx {
            self.descriptors.resize(idx + 1, None);
            self.aliases.resize(idx + 1, None);
        }
        if let Some(previous) = self.descriptors[idx] {
            assert_eq!(
                previous, desc,
                "GPU field id {:?} was registered with conflicting partner metadata",
                field_id,
            );
            return;
        }

        let canonical = match desc.buffer_key {
            Some(key) => match self.named.get(key).copied() {
                Some(existing_id) => {
                    let existing = self
                        .descriptor(existing_id)
                        .expect("named GPU partner has no canonical descriptor");
                    assert_eq!(
                        existing.value_token.desc(),
                        desc.value_token.desc(),
                        "incompatible #[gpu(buffer = \"{key}\")] row layout/stride: existing {:?}, new {:?}",
                        existing.value_token.desc(), desc.value_token.desc(),
                    );
                    assert_eq!(
                        existing.value_token, desc.value_token,
                        "incompatible #[gpu(buffer = \"{key}\")] row types: existing {:?}, new {:?}",
                        existing.value_token, desc.value_token,
                    );
                    assert_eq!(
                        existing.mode, desc.mode,
                        "incompatible #[gpu(buffer = \"{key}\")] mirror modes: existing {:?}, new {:?}",
                        existing.mode, desc.mode,
                    );
                    existing_id
                }
                None => {
                    self.named.insert(key, field_id);
                    field_id
                }
            },
            None => field_id,
        };

        self.aliases[idx] = Some(canonical);
        self.descriptors[idx] = Some(desc);
    }

    fn claim_residency(&mut self, field_id: ComponentId, residency: GpuBufferResidency) {
        let canonical = self.resolve(field_id);
        let idx = canonical.0 as usize;
        if self.residencies.len() <= idx {
            self.residencies.resize(idx + 1, None);
        }
        if let Some(previous) = self.residencies[idx] {
            assert_eq!(
                previous, residency,
                "GPU partner {:?} was registered for incompatible {:?} and {:?} allocation domains; fixed CellStorage and growable World storage cannot share one physical allocation in the same SceneGpuStore",
                canonical, previous, residency,
            );
        } else {
            self.residencies[idx] = Some(residency);
        }
    }

    fn claim_world_owner(&mut self, owner: ComponentId, field_id: ComponentId) {
        let canonical = self.resolve(field_id);
        let idx = canonical.0 as usize;
        if self.world_owners.len() <= idx {
            self.world_owners.resize(idx + 1, None);
        }
        if let Some(previous) = self.world_owners[idx] {
            assert_eq!(
                previous, owner,
                "World GPU buffer {:?} is shared by component owners {:?} and {:?}; \
                 compatible named-buffer aliasing is supported for disjoint CellStorage \
                 regions, but World components have independent local-row domains and would \
                 race with last-write-wins semantics. Give each World component a distinct \
                 #[gpu(buffer = ...)] key or consolidate the row into one component",
                canonical, previous, owner,
            );
        } else {
            self.world_owners[idx] = Some(owner);
        }
    }

    #[inline]
    fn world_owner(&self, field_id: ComponentId) -> Option<ComponentId> {
        let canonical = self.resolve(field_id);
        self.world_owners
            .get(canonical.0 as usize)
            .and_then(|owner| *owner)
    }
}

/// Trait for GPU-mirrored component types. Generated by
/// `#[derive(SceneStore)]`.
///
/// # Safety
///
/// Every descriptor must identify a fully initialized, no-padding Pod field
/// within `Self`; offsets and value tokens must match the actual field type.
/// Generated implementations enforce this through bytemuck on each `#[gpu]`
/// field wrapper. The whole component is not itself treated as raw bytes.
pub unsafe trait GpuColumnSet: 'static {
    /// Describe each `#[gpu]` field in declaration order.
    fn gpu_columns() -> Vec<GpuColumnDesc>;

    /// Write a component instance's GPU fields to the store + cell.
    /// Called by `SceneGpuStore::write_gpu()`.
    fn write_gpu(
        store: &SceneGpuStore,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        data: &Self,
        phase: &impl SimulateWitness,
    );
}

/// Construction-time registration contract for a component whose GPU partner
/// columns use the growable World-mirror storage model.
///
/// `#[derive(SceneStore)]` implements this trait only for concrete types that
/// contain at least one `#[gpu]` field. Keeping registration behind a trait
/// lets higher-level integrations expose a typed registrar without lending out
/// the entire [`SceneGpuStore`] and its ambient access to unrelated buffers.
/// CPU-only derives intentionally do not implement this trait.
///
/// # Safety
///
/// Implementations receive construction-time access to a whole
/// [`SceneGpuStore`]. They must register only `Self`'s reflected descriptors,
/// component ownership, presence column, and matching growable buffers. They
/// must not inspect, mutate, alias, or retain handles to any unrelated partner.
/// Prefer the implementation generated by `#[derive(SceneStore)]`.
pub unsafe trait GrowableGpuColumnSet: GpuColumnSet {
    /// Register every reflected partner column plus the component-local
    /// presence buffer before a World mirror is attached.
    fn register_gpu_columns_growable(
        store: &mut SceneGpuStore,
        initial_capacity: u32,
        device: &std::sync::Arc<wgpu::Device>,
    );
}

/// Opaque handle to a registered cell's region assignment. Indexes
/// `SceneGpuStore`'s internal per-cell state; never crosses the FFI/shader
/// boundary (that's `row_region_base`'s job).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId(pub(crate) u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Write,
    Retired,
    Compacted,
}

pub(crate) struct QueuedRetire {
    pub(crate) pending: PendingRetire,
    pub(crate) serial: u64,
}

/// Per-cell GPU-side bookkeeping: the region assignment, the dirty state
/// that used to live on `GpuStore` directly (M2a), and the deferred-retire
/// queue (now one per cell rather than store-wide).
pub(crate) struct CellGpuState {
    /// Size class this cell was registered under (M2b-β `unregister_cell`:
    /// selects which `row_pools`/`slot_pools` entry to return the region
    /// pair to at eviction).
    pub(crate) class: usize,
    pub(crate) row_base: u32,
    pub(crate) slot_base: u32,
    /// Class capacity + headroom; bounds every gen/slot write into this
    /// cell's region.
    pub(crate) slot_capacity: u32,
    /// Per-column dirty masks in a DENSE table indexed by `ComponentId.0`,
    /// replacing the old named `dirty_transforms` / `dirty_infos` fields
    /// (pre-work item 2). The slot-mirror dirty mask (`dirty_slots`) is kept
    /// separate because it is driven by the self-healing boundary scan, not
    /// by column writes.
    ///
    /// **Why dense, not a `HashMap`:** every mirrored write (`write_transform`,
    /// `write_instance_info`, and every derive-generated `write_gpu`) marks a
    /// row through this table, so it sits in the hottest path the crate has.
    /// `ComponentId`s are dense and allocated from 1 (`component::component_id`),
    /// so a `Vec` index replaces a hash of the key on every single write.
    /// Measured on the delta-vs-legacy head-to-head (`legacy_model_bench`,
    /// 2 runs): the hashed form cost ~+15% CPU at 1% mutation and ~+50% at
    /// 100% mutation versus the named-field form it generalized. Index 0 is
    /// always `None` (`ComponentId` 0 is reserved); the table is sized to the
    /// highest registered GPU-column id at `register_cell`.
    pub(crate) dirty_columns: Vec<Option<DirtyMask>>,
    /// CPU-column id feeding each canonical GPU buffer id. Usually the two
    /// ids are equal; an explicit named partner may alias a component-local
    /// wrapper id to an earlier compatible physical-buffer id.
    column_sources: Vec<Option<ComponentId>>,
    /// Reflection entries for GPU columns this particular cell actually
    /// carries, retained in CellStorage declaration order. Populated once at
    /// registration so editor/debug queries allocate only their return Vec.
    gpu_columns: Vec<GpuColumnDesc>,
    dirty_slots: DirtyMask,
    /// Per-row global-slot staging. `sync_all`'s self-healing boundary scan
    /// is scan-first, not dirty-mask-first: it compares `slot_shadow` against
    /// the authoritative slot column for every occupied row, and for each
    /// mismatch it fills this scratch entry AND marks `dirty_slots` together
    /// (neither drives the other) before uploading into the shared
    /// slot-mirror SSBO (T4; C6 GPU handle validation).
    slot_scratch: Vec<u32>,
    /// Per-ROW shadow of the last LOCAL slot uploaded into the mirror for
    /// that row; `u32::MAX` = never uploaded. Read and written ONLY by
    /// `sync_all`'s self-healing boundary scan (`&mut self`), which compares
    /// it against the authoritative slot column and re-uploads every
    /// mismatch. Row-scoped on purpose: per-event triggers (gen-gate,
    /// write-path shadow check, compaction marks) each missed a staleness
    /// path — e.g. an alloc re-occupying a compaction-vacated row that is
    /// never written (Task 4 review + re-review); the boundary scan closes
    /// them all with one invariant.
    slot_shadow: Vec<u32>,
    /// Per-cell deferred-retire queue; nondecreasing serials (debug-asserted,
    /// T11).
    pub(crate) pending: VecDeque<QueuedRetire>,
    /// CPU-side shadow of the last generation uploaded per LOCAL slot (§4
    /// delta-minimality on the write path), seeded from the registry at
    /// `register_cell`. Atomic because `write_transform` takes `&self`.
    gen_shadow: Vec<AtomicU32>,
}

impl CellGpuState {
    /// The dirty mask for one mirrored column, or `None` if this cell has no
    /// such column registered. One bounds-checked index — no hashing (see
    /// [`CellGpuState::dirty_columns`]).
    #[inline]
    fn dirty_column(&self, id: ComponentId) -> Option<&DirtyMask> {
        self.dirty_columns
            .get(id.0 as usize)
            .and_then(Option::as_ref)
    }

    /// Every registered column's `(id, mask)` pair, skipping the dense
    /// table's holes. Frame-boundary use (compact/sync), not the write path.
    #[inline]
    fn dirty_columns_iter(&self) -> impl Iterator<Item = (ComponentId, ComponentId, &DirtyMask)> {
        self.dirty_columns.iter().enumerate().filter_map(|(i, m)| {
            m.as_ref().map(|mask| {
                let buffer_id = ComponentId(i as u32);
                let column_id =
                    self.column_sources[i].expect("dirty GPU column has no CellStorage source id");
                (buffer_id, column_id, mask)
            })
        })
    }
}

/// One cell's region assignment paired with its mutable storage, for the
/// bulk `*_all` frame-boundary stages.
///
/// The (id, cell) pairing is TRUSTED: the store cannot verify that `cell` is
/// the storage `id` was registered with, and a mismatched pair commits
/// retires and dirty marks into the wrong cell's regions.
pub struct CellSlot<'a> {
    pub id: CellId,
    pub cell: &'a mut CellStorage,
}

/// The SceneDB-owned multi-cell device-side store (M2b-α §2): persistent
/// region-partitioned scene SSBOs, the mirrored-column writer, delta-sync,
/// and the retirement drain — generalizing M2a's single-cell `GpuStore` to
/// N cells sharing one set of buffers via `RegionPool` (§7).
pub struct SceneGpuStore {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    dirty_tracked_reallocation_policy: DirtyTrackedReallocationPolicy,
    /// Stable field↔buffer partner metadata. Kept separate from the physical
    /// allocation maps so fixed CellStorage and growable World stores use the
    /// same identity and compatibility rules without conflating their
    /// different capacity/lifetime models.
    gpu_partners: GpuPartnerRegistry,
    /// Concrete World dispatch installed by the same explicit registration
    /// call that allocates each component's partner columns. Link-time
    /// inventory is retained only as a hosted compatibility fallback.
    world_mirror_registrations: HashMap<ComponentId, RegistrationFns>,
    /// Type-erased GPU buffers keyed by `ComponentId`.  Replaces the old
    /// concrete `transforms` / `instance_infos` fields (pre-work item 3).
    /// Registered via [`Self::register_gpu_buffer`].
    gpu_buffers: HashMap<ComponentId, Box<dyn GpuBufferDispatch>>,
    /// Growable counterpart to `gpu_buffers`, disjoint from it — a given
    /// `ComponentId` is registered in exactly one of the two maps, never
    /// both. Registered via [`Self::register_growable_gpu_buffer`]; see
    /// `gpu::growable_scene_buffer`'s module docs for why this needs its own
    /// map/trait rather than living in `gpu_buffers` (`GpuBufferDispatch::
    /// buffer(&self) -> &wgpu::Buffer` can't be implemented soundly once the
    /// buffer lives behind the `RwLock` growth requires).
    growable_gpu_buffers: HashMap<ComponentId, Box<dyn GrowableGpuBufferDispatch>>,
    /// Dirty-tracked counterpart to `growable_gpu_buffers` -- disjoint from
    /// it, `gpu_buffers`, and `once_gpu_buffers`; a field `ComponentId`
    /// lives in exactly one data-buffer map. Registered via
    /// [`Self::register_dirty_tracked_gpu_buffer`], for `#[gpu(mirror =
    /// DirtyTracked)]` World-mirrored fields (the default `#[gpu]` mode) --
    /// writes are marked dirty ([`Self::mark_gpu_row_dirty`]) rather than
    /// uploaded immediately; [`Self::flush_gpu_mirror`] performs the actual
    /// coalesced upload.
    dirty_tracked_gpu_buffers: HashMap<ComponentId, Box<dyn DirtyTrackedGpuBufferDispatch>>,
    /// Explicit one-time-handoff columns. These retain the GPU allocation
    /// and only a transient pending queue between handoff and flush -- no
    /// full-capacity CPU value shadow. See `gpu::once_scene_buffer`.
    once_gpu_buffers: HashMap<ComponentId, Box<dyn OnceGpuBufferDispatch>>,
    /// One presence/tombstone column per GPU-bearing *component type*, keyed
    /// by the component's id (not any individual field-buffer id). Entity
    /// generations cannot represent component removal while the entity
    /// remains alive, so shaders validate both dimensions.
    component_presence_buffers: HashMap<ComponentId, ComponentPresenceBuffer>,
    slot_mirror: SceneBuffer<u32>,
    generations: GenerationBuffer,
    // `material` (32-byte placeholder buffer + `material_buffer()` accessor)
    // retired M3-α T11 (Rev 2.4 R8, approved 2026-07-16): the real 64-byte
    // material row (`gpu::MaterialRow`) is owned by the standalone
    // `gpu::MaterialRegistry`, mirroring `MeshRegistry`'s shape — not by
    // this store. The placeholder was never written to by anything and
    // predated R8's row layout by two size classes (32 B here vs. 64 B
    // there); keeping both would leave two unrelated "material buffer"
    // concepts in the crate (Ownership Law, CONTRACTS C0: one clear owner
    // per buffer).
    cell_metadata: wgpu::Buffer,
    tracker: SubmissionTracker,
    phase: Phase,
    /// One row pool and one slot pool per size class, base offsets laid end
    /// to end in class order (§7).
    row_pools: Vec<RegionPool>,
    slot_pools: Vec<RegionPool>,
    /// One slot per ever-registered `CellId`. `None` means the cell was
    /// evicted (`unregister_cell`, M2b-β §4.1) — `CellId`s are NOT recycled,
    /// so a hole here is permanent for the life of the store; a fresh
    /// registration always pushes a new `Some(..)` at the end regardless of
    /// how many holes precede it.
    cells: Vec<Option<CellGpuState>>,
    /// Instrumentation: total generation-buffer writes issued across every
    /// cell, so tests can assert generation-upload minimality.
    gen_writes: AtomicU64,
}

impl SceneGpuStore {
    /// Cheap `Arc` clone of the device this store was constructed with.
    /// Needed by anything that must allocate a NEW buffer alongside this
    /// store's own (e.g. [`crate::gpu::world_mirror::GenerationMirror`],
    /// which owns a device-independent growable buffer rather than one
    /// registered in this store's own `gpu_buffers`/`growable_gpu_buffers`
    /// maps) without threading a separate `Arc<wgpu::Device>` through every
    /// call site that already has a `&SceneGpuStore` in hand.
    pub fn device_arc(&self) -> Arc<wgpu::Device> {
        Arc::clone(&self.device)
    }

    /// Select the construction-time reallocation policy for subsequently
    /// registered DirtyTracked World columns and their liveness buffers.
    pub fn set_dirty_tracked_reallocation_policy(
        &mut self,
        policy: DirtyTrackedReallocationPolicy,
    ) {
        assert!(
            self.dirty_tracked_gpu_buffers.is_empty()
                && self.component_presence_buffers.is_empty(),
            "DirtyTracked reallocation policy must be selected before World GPU columns are registered",
        );
        self.dirty_tracked_reallocation_policy = policy;
    }

    pub fn dirty_tracked_reallocation_policy(&self) -> DirtyTrackedReallocationPolicy {
        self.dirty_tracked_reallocation_policy
    }

    pub fn new(ctx: &EngineGpuContext, cfg: SceneGpuConfig) -> Self {
        let mut row_pools = Vec::with_capacity(cfg.classes.len());
        let mut slot_pools = Vec::with_capacity(cfg.classes.len());
        let mut row_offset = 0u32;
        let mut slot_offset = 0u32;
        for class in &cfg.classes {
            let slot_region_size = class.capacity + cfg.tombstone_headroom;
            row_pools.push(RegionPool::new(
                row_offset,
                class.capacity,
                class.max_resident_cells,
            ));
            slot_pools.push(RegionPool::new(
                slot_offset,
                slot_region_size,
                class.max_resident_cells,
            ));
            // Checked accumulation: a pathological config must fail loudly at
            // construction, not wrap into silently-undersized SSBOs.
            row_offset = row_offset
                .checked_add(
                    class
                        .capacity
                        .checked_mul(class.max_resident_cells)
                        .expect("row capacity overflow"),
                )
                .expect("row capacity overflow");
            slot_offset = slot_offset
                .checked_add(
                    slot_region_size
                        .checked_mul(class.max_resident_cells)
                        .expect("slot capacity overflow"),
                )
                .expect("slot capacity overflow");
        }
        let mut store = Self {
            device: Arc::clone(ctx.device()),
            queue: Arc::clone(ctx.queue()),
            dirty_tracked_reallocation_policy: DirtyTrackedReallocationPolicy::GpuCopy,
            gpu_partners: GpuPartnerRegistry::default(),
            world_mirror_registrations: HashMap::new(),
            gpu_buffers: HashMap::new(),
            growable_gpu_buffers: HashMap::new(),
            dirty_tracked_gpu_buffers: HashMap::new(),
            once_gpu_buffers: HashMap::new(),
            component_presence_buffers: HashMap::new(),
            slot_mirror: SceneBuffer::new(ctx.device(), "scenedb-slot-mirror", row_offset),
            generations: GenerationBuffer::new(ctx.device(), slot_offset),
            // Per-cell metadata stride is 8 bytes (design §4.1: f32 alpha +
            // u32 domain). Allocated at final stride now (§10); α has no
            // writer.
            cell_metadata: ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("scenedb-cell-metadata"),
                size: cfg.max_cells_metadata as u64 * 8,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            tracker: SubmissionTracker::new(),
            phase: Phase::Write,
            row_pools,
            slot_pools,
            cells: Vec::new(),
            gen_writes: AtomicU64::new(0),
        };
        // Register built-in partner metadata before their buffers, exactly
        // like derive-generated component registration. This makes ordinary
        // SpatialCell transform/instance columns visible to the same generic
        // reflection and snapshot APIs as custom `#[gpu]` fields.
        store.register_gpu_column_descs([
            GpuColumnDesc {
                field_token: crate::token::TypeToken::of::<[f32; 16]>(),
                value_token: crate::token::TypeToken::of::<[f32; 16]>(),
                field_offset: 0,
                mode: MirrorMode::DirtyTracked,
                buffer_name: "transform",
                buffer_key: None,
            },
            GpuColumnDesc {
                field_token: crate::token::TypeToken::of::<InstanceInfo>(),
                value_token: crate::token::TypeToken::of::<InstanceInfo>(),
                field_offset: 0,
                mode: MirrorMode::DirtyTracked,
                buffer_name: "instance_info",
                buffer_key: None,
            },
        ]);
        // Register built-in GPU buffers.
        store.register_gpu_buffer::<[f32; 16]>(row_offset, ctx.device(), "scenedb-instances");
        store.register_gpu_buffer::<InstanceInfo>(
            row_offset,
            ctx.device(),
            "scenedb-instance-info",
        );
        store
    }

    pub fn tracker(&self) -> &SubmissionTracker {
        &self.tracker
    }

    /// Register reflection and identity metadata for GPU partner columns.
    /// Explicit keys are interned before their typed buffers are allocated;
    /// compatible later fields then reuse the first physical buffer.
    pub fn register_gpu_column_descs(&mut self, descs: impl IntoIterator<Item = GpuColumnDesc>) {
        for desc in descs {
            let field_id = desc.field_token.id();
            if let Some(key) = desc.buffer_key {
                if let Some(canonical) = self.gpu_partners.id_for_key(key) {
                    let field_has_allocation = self.gpu_buffers.contains_key(&field_id)
                        || self.growable_gpu_buffers.contains_key(&field_id)
                        || self.dirty_tracked_gpu_buffers.contains_key(&field_id)
                        || self.once_gpu_buffers.contains_key(&field_id);
                    assert!(
                        canonical == field_id || !field_has_allocation,
                        "#[gpu(buffer = \"{key}\")] metadata was registered after field {:?} already allocated its own GPU buffer; register descriptors before typed buffers",
                        field_id,
                    );
                }
            }
            self.gpu_partners.register(desc);
        }
    }

    /// Convenience for hand-written [`GpuColumnSet`] implementations.
    pub fn register_gpu_columns_for<T: GpuColumnSet>(&mut self) {
        self.register_gpu_column_descs(T::gpu_columns());
    }

    /// Couple a concrete component's World mirror dispatch to the explicit
    /// growable-buffer registration that owns its partner columns.
    #[doc(hidden)]
    pub fn register_world_mirror_registration(&mut self, registration: GpuMirrorRegistration) {
        let id = (registration.component_id)();
        let previous = self.world_mirror_registrations.insert(
            id,
            RegistrationFns {
                dispatch: registration.dispatch,
                clear: registration.clear,
                descriptors: registration.descriptors,
            },
        );
        assert!(
            previous.is_none(),
            "World mirror component {id:?} was explicitly registered more than once"
        );
    }

    #[inline]
    pub(crate) fn world_mirror_registration(
        &self,
        id: ComponentId,
    ) -> Option<RegistrationFns> {
        self.world_mirror_registrations.get(&id).copied()
    }

    /// Claims one canonical GPU destination for a single owning World
    /// component. Generated growable registration calls this after partner
    /// descriptors have resolved explicit aliases and before allocating the
    /// physical buffer.
    ///
    /// The same owner may claim several distinct destinations. Two owners
    /// may not claim aliases of the same destination: unlike CellStorage's
    /// disjoint row regions, their independent component-local allocators
    /// would write unrelated entities to the same physical rows.
    #[doc(hidden)]
    pub fn register_world_gpu_column_owner(
        &mut self,
        owner_component_id: ComponentId,
        field_id: ComponentId,
    ) {
        self.gpu_partners
            .claim_world_owner(owner_component_id, field_id);
    }

    /// Resolve a component-local field id to the canonical physical buffer
    /// id. This is a dense-vector lookup on the per-row World write path.
    #[inline]
    pub fn resolve_gpu_buffer_id(&self, id: ComponentId) -> ComponentId {
        self.gpu_partners.resolve(id)
    }

    /// Resolve an explicit `#[gpu(buffer = "...")]` key to its canonical
    /// physical buffer id.
    pub fn gpu_buffer_id_for_key(&self, key: &str) -> Option<ComponentId> {
        self.gpu_partners.id_for_key(key)
    }

    /// Register a GPU buffer for a Pod column type.  Called automatically
    /// for `[f32; 16]` (transforms) and `InstanceInfo` during [`Self::new`];
    /// derive macros for custom SceneComponent types will call this during
    /// store construction.
    pub fn register_gpu_buffer<T: Pod + Send + Sync + HasTypeToken + 'static>(
        &mut self,
        capacity: u32,
        device: &wgpu::Device,
        label: &str,
    ) {
        let field_id = <T as HasTypeToken>::type_token().id();
        self.gpu_partners
            .claim_residency(field_id, GpuBufferResidency::FixedCell);
        let id = self.gpu_partners.resolve(field_id);
        assert!(
            !self.growable_gpu_buffers.contains_key(&id)
                && !self.dirty_tracked_gpu_buffers.contains_key(&id)
                && !self.once_gpu_buffers.contains_key(&id),
            "GPU partner {:?} is already registered with a World allocation; fixed CellStorage and growable World buffers cannot share one physical allocation in the same SceneGpuStore",
            id,
        );
        if let Some(existing) = self.gpu_buffers.get(&id) {
            assert_eq!(
                existing.element_size(),
                std::mem::size_of::<T>(),
                "compatible named GPU partners must have the same physical stride",
            );
            assert_eq!(
                existing.capacity(),
                capacity,
                "compatible named fixed GPU partners must use the same capacity",
            );
            return;
        }
        let buffer = SceneBuffer::<T>::new(device, label, capacity);
        self.gpu_buffers.insert(id, Box::new(buffer));
    }

    /// Growable counterpart to [`Self::register_gpu_buffer`] — for
    /// World-mirrored `#[gpu]` columns, whose required capacity isn't known
    /// ahead of time (`World`'s entity count has no ceiling). `initial_capacity`
    /// only needs to be cheap to allocate, not sized for the eventual world —
    /// the buffer grows (doubling, GPU-to-GPU copy) transparently on writes
    /// past its current capacity, via [`Self::write_row_bytes_growing`].
    ///
    /// `max_capacity: None` (the recommendation for World-mirrored columns,
    /// and what `#[derive(SceneStore)]`'s generated `register_gpu_columns_growable`
    /// always passes) means growth can never fail — exactly the property
    /// that makes this safe to call from `World::insert`'s automatic
    /// dispatch path with no `Result` to propagate through `insert` itself.
    /// Pass `Some(max)` only for a deliberate, caller-managed VRAM ceiling
    /// (not recommended for World-mirrored columns specifically, since
    /// hitting it turns an ordinary `world.insert()` call into one that can
    /// fail with no way to report that failure back to the caller — see
    /// [`Self::write_row_bytes_growing`]'s doc).
    ///
    /// A given `ComponentId` must be registered through exactly one of
    /// [`Self::register_gpu_buffer`]/`Self::register_growable_gpu_buffer` —
    /// registering the same id both ways leaves the growable one shadowed by
    /// [`Self::write_row_bytes`]'s lookup order and is almost certainly not
    /// what was intended; this is not currently asserted against (no test
    /// covers the double-registration case), so avoid it by construction.
    pub fn register_growable_gpu_buffer<T: Pod + Send + Sync + HasTypeToken + 'static>(
        &mut self,
        initial_capacity: u32,
        max_capacity: Option<u32>,
        device: &Arc<wgpu::Device>,
        label: &str,
    ) {
        let field_id = <T as HasTypeToken>::type_token().id();
        self.gpu_partners
            .claim_residency(field_id, GpuBufferResidency::GrowableWorld);
        let id = self.gpu_partners.resolve(field_id);
        assert!(
            !self.gpu_buffers.contains_key(&id)
                && !self.dirty_tracked_gpu_buffers.contains_key(&id)
                && !self.once_gpu_buffers.contains_key(&id),
            "GPU partner {:?} is already registered through a different allocation path",
            id,
        );
        if let Some(existing) = self.growable_gpu_buffers.get(&id) {
            assert_eq!(
                existing.element_size(),
                std::mem::size_of::<T>(),
                "compatible named GPU partners must have the same physical stride",
            );
            return;
        }
        let buffer = GrowableSceneBuffer::<T>::new(
            Arc::clone(device),
            label,
            initial_capacity,
            max_capacity,
        );
        self.growable_gpu_buffers.insert(id, Box::new(buffer));
    }

    /// Mark a column's row as dirty for the next GPU sync.
    /// Called by the derive macro's generated `write_gpu()`.
    ///
    /// # Panics
    ///
    /// Panics if `id` has not been registered.  Silently ignores unknown
    /// component ids (the column was registered via `register_gpu_buffer`,
    /// and every `#[gpu]` field's type gets one).
    pub fn mark_column_dirty(&self, id: CellId, component_id: ComponentId, row: u32) {
        let state = self.cells[id.0 as usize]
            .as_ref()
            .expect("cell unregistered");
        let component_id = self.gpu_partners.resolve(component_id);
        if let Some(mask) = state.dirty_column(component_id) {
            mask.mark(row);
        }
    }

    /// Generic write path for any GPU-mirrored component type.
    /// Writes data to the CPU columns via the trait's `write_gpu`, marks dirty
    /// for GPU sync, and stamps the generation.
    ///
    /// Returns `false` for stale/invalid handles or pinned rows.
    pub fn write_gpu<T: GpuColumnSet>(
        &self,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        data: &T,
        phase: &impl SimulateWitness,
    ) -> bool {
        debug_assert_eq!(
            self.phase,
            Phase::Write,
            "mutation outside the write window"
        );
        let state = self.cells[id.0 as usize]
            .as_ref()
            .expect("cell unregistered");
        let Some(row) = cell.row_of(handle) else {
            return false;
        };
        if cell.is_row_pinned(row) {
            return false;
        }
        // Delegate to the trait's write_gpu — writes CPU columns + marks dirty
        T::write_gpu(self, id, cell, handle, data, phase);
        // Stamp generation (shadow-gated, idempotent)
        self.write_generation(state, handle.index(), handle.generation());
        true
    }

    /// Test instrument: how many generation-buffer writes this store has
    /// issued across every cell (asserting upload minimality, §4).
    ///
    /// Counts only `write_generation` calls (per-slot, shadow-gated). The
    /// recycled-region tail scrub in `register_cell` (§11 D2-tail
    /// carry-forward) bypasses this counter deliberately: it is a
    /// region-lifecycle bulk write (once per promotion, not per slot), not
    /// the per-slot stamp this instrument exists to bound.
    #[doc(hidden)]
    pub fn generation_write_count(&self) -> u64 {
        self.gen_writes.load(Ordering::Relaxed)
    }

    /// Shadow-gated generation upload: writes VRAM (and the shadow) only when
    /// `generation` differs from the last value uploaded for `local_slot`,
    /// translated to the cell's global slot via `state.slot_base`.
    ///
    /// Deliberately NOT the slot-mirror dirty trigger: this gate is
    /// SLOT-scoped, but mirror staleness is ROW-scoped — a retired slot
    /// recycled into a different row arrives with its generation already
    /// shadowed (the retire stamped it), so the gate stays silent while the
    /// new row's mirror entry is stale (fail-open C6, Task 4 review). The
    /// mirror trigger is `sync_all`'s self-healing boundary scan.
    fn write_generation(&self, state: &CellGpuState, local_slot: u32, generation: u32) {
        assert!(
            local_slot < state.slot_capacity,
            "slot {local_slot} beyond region capacity {} — write must never land in a neighbor's region",
            state.slot_capacity
        );
        if state.gen_shadow[local_slot as usize].load(Ordering::Relaxed) == generation {
            return;
        }
        self.generations
            .write(&self.queue, state.slot_base + local_slot, generation);
        state.gen_shadow[local_slot as usize].store(generation, Ordering::Relaxed);
        self.gen_writes.fetch_add(1, Ordering::Relaxed);
    }

    /// §4.1 promotion primitive (α: registration; β reuses it for promotion):
    /// allocates row+slot regions, bulk-rebuilds the generation region from
    /// the registry, seeds the gen-shadow, marks all occupied rows dirty in
    /// the transform mask. The slot mirror needs no warm-up — the first
    /// `sync_all` boundary scan uploads every occupied row's slot entry.
    ///
    /// **Panic vs. decline:** the two failure modes below are deliberately
    /// different in kind. Pool exhaustion (`RowsExhausted`/`SlotsExhausted`)
    /// is an ordinary runtime condition — a graceful `Err` the executor
    /// declines on (the cell stays in its current domain, §8) — because a
    /// full grid is expected, not a bug. The `assert!`s below it (cell rows/
    /// slots exceeding the class's region capacity) are invariant violations:
    /// a cell that does not fit the size class it was registered under is a
    /// caller/config bug, not a transient condition, so those panic by
    /// design rather than returning `Err`.
    pub fn register_cell(
        &mut self,
        cell: &CellStorage,
        class: usize,
    ) -> Result<CellId, RegionError> {
        if self.row_pools[class].free_count() == 0 {
            return Err(RegionError::RowsExhausted);
        }
        if self.slot_pools[class].free_count() == 0 {
            return Err(RegionError::SlotsExhausted);
        }
        let row_base = self.row_pools[class]
            .alloc()
            .expect("checked free_count above");
        let slot_base = self.slot_pools[class]
            .alloc()
            .expect("checked free_count above");
        let row_capacity = self.row_pools[class].region_size();
        let slot_capacity = self.slot_pools[class].region_size();

        assert!(
            cell.rows_in_use() <= row_capacity,
            "cell occupies {} rows but class capacity is {row_capacity}",
            cell.rows_in_use()
        );
        let gens = cell.registry().generations();
        assert!(
            gens.len() as u32 <= slot_capacity,
            "cell has more slots ({}) than its region capacity {slot_capacity}",
            gens.len()
        );
        self.generations
            .rebuild_region(&self.queue, slot_base, gens);

        // Tail scrub (§11 D2-tail carry-forward, binding for β): `gens`
        // covers only the registry's allocated-slot prefix, never the full
        // region. On a FRESH region the tail is already zero (never
        // written), but on a RECYCLED region (post-eviction, `unregister_cell`)
        // the tail still holds the prior tenant's VRAM generations — an
        // allocated-but-never-written slot in the new tenant would fail
        // fail-open, validating against a stranger's stale generation. Zero
        // the tail unconditionally; the write is a no-op in cost terms only
        // when `occupied == slot_capacity` (headroom-free region). Cold
        // path: once per promotion, not per slot — deliberately NOT counted
        // by `gen_writes` (see that counter's doc).
        let occupied = gens.len() as u32;
        if occupied < slot_capacity {
            let tail_zeros = vec![0u32; (slot_capacity - occupied) as usize];
            self.queue.write_buffer(
                self.generations.buffer(),
                (slot_base + occupied) as u64 * 4,
                super::as_bytes(&tail_zeros),
            );
        }

        // gen_shadow: 0 by construction for every slot (including the tail
        // computed above — a recycled region's shadow must forget the prior
        // tenant's shadow state too, not just its VRAM bytes), then seeded
        // for the occupied prefix from the registry.
        let gen_shadow: Vec<AtomicU32> = (0..slot_capacity).map(|_| AtomicU32::new(0)).collect();
        for (slot, &generation) in gens.iter().enumerate() {
            gen_shadow[slot].store(generation, Ordering::Relaxed);
        }

        // Match the cell's actual CPU columns to registered fixed GPU
        // buffers. This is also where a component-local wrapper id is joined
        // to the canonical id of an explicitly named partner. Building masks
        // only for columns the cell carries avoids the old O(all globally
        // registered GPU columns) dirty-mask overhead on every cell.
        let matched_columns: Vec<(ComponentId, ComponentId)> = cell
            .token_index_slice()
            .iter()
            .filter_map(|&(column_id, _)| {
                let buffer_id = self.gpu_partners.resolve(column_id);
                self.gpu_buffers
                    .contains_key(&buffer_id)
                    .then_some((column_id, buffer_id))
            })
            .collect();
        let widest = matched_columns
            .iter()
            .map(|&(_, id)| id.0 as usize)
            .max()
            .unwrap_or(0);
        let mut dirty_columns: Vec<Option<DirtyMask>> = Vec::new();
        dirty_columns.resize_with(widest + 1, || None);
        let mut column_sources = vec![None; widest + 1];
        let mut gpu_columns = Vec::new();
        for (column_id, buffer_id) in matched_columns {
            assert!(
                dirty_columns[buffer_id.0 as usize].is_none(),
                "cell contains more than one CPU column mapped to GPU partner {:?}; one physical row cannot have two authoritative sources",
                buffer_id,
            );
            let mask = DirtyMask::new(row_capacity);
            mask.mark_range(cell.rows_in_use());
            dirty_columns[buffer_id.0 as usize] = Some(mask);
            column_sources[buffer_id.0 as usize] = Some(column_id);
            if let Some(desc) = self.gpu_partners.descriptor(column_id) {
                gpu_columns.push(desc);
            }
        }
        // No slot-mirror warm-up: `sync_all`'s self-healing boundary scan is
        // the SOLE dirty_slots marker and scratch/shadow writer. The shadow
        // starts all-MAX (= never uploaded; real local slots are always
        // < slot_capacity < u32::MAX), so the first boundary marks and
        // uploads every occupied row on its own. Keeping mark/fill paired in
        // exactly one place removes the "mark without a scratch fill uploads
        // stale bytes" footgun.
        let dirty_slots = DirtyMask::new(row_capacity);

        self.cells.push(Some(CellGpuState {
            class,
            row_base,
            slot_base,
            slot_capacity,
            dirty_columns,
            column_sources,
            gpu_columns,
            dirty_slots,
            slot_scratch: vec![0u32; row_capacity as usize],
            slot_shadow: vec![u32::MAX; row_capacity as usize],
            pending: VecDeque::new(),
            gen_shadow,
        }));
        Ok(CellId(self.cells.len() as u32 - 1))
    }

    /// Test 14 (C0 companion gate): build a fresh multi-cell store on a fresh
    /// device purely from every cell's CPU-authoritative columns (no GPU-only
    /// state exists to lose, design §3 "derived data is not stored"). Returns
    /// the rebuilt store paired with each input cell's freshly assigned
    /// `CellId`, in the same order as `cells`.
    ///
    /// Precondition per cell: no rows may be pinned (all pending retires
    /// drained via `retire_all`) — recovery of in-flight retirement across a
    /// device loss is M4 scope; rebuilding while a pin is outstanding would
    /// strand it permanently (the pin bit lives only in `CellStorage`, and
    /// this fresh store has no queued `PendingRetire` to eventually unpin
    /// it). Verbatim message carried over from M2a's `GpuStore::rebuild_from`.
    ///
    /// For each cell: `register_cell` already rebuilds that cell's generation
    /// region, seeds the gen shadow, and marks every occupied row dirty in
    /// the transform mask (§4.1 warm-up) — but `write_rows` below is an
    /// UNCONDITIONAL bulk write, so those warm-up marks are cleared right
    /// after to avoid double-uploading the same bytes at the first boundary.
    /// The slot mirror has no warm-up marker of its own (its sole dirty
    /// trigger is `sync_all`'s self-healing boundary scan, which hasn't run
    /// yet for a freshly rebuilt store), so it is bulk-filled here too —
    /// scratch and shadow alike — matching exactly what that boundary scan
    /// would otherwise produce on its first pass.
    pub fn rebuild(
        ctx: &EngineGpuContext,
        cfg: SceneGpuConfig,
        cells: &[(usize, &CellStorage)],
    ) -> (Self, Vec<CellId>) {
        let mut store = Self::new(ctx, cfg);
        let mut ids = Vec::with_capacity(cells.len());
        for &(class, cell) in cells {
            debug_assert!(
                (0..cell.rows_in_use()).all(|r| !cell.is_row_pinned(r)),
                "rebuild_from with in-flight retirement: drain retire() before device-loss rebuild — pins would be permanently stranded"
            );
            let id = store
                .register_cell(cell, class)
                .expect("rebuild: cell must fit its class region");
            let rows = cell.rows_in_use();
            let row_base = store.cells[id.0 as usize]
                .as_ref()
                .expect("cell unregistered")
                .row_base;
            let slot_base = store.cells[id.0 as usize]
                .as_ref()
                .expect("cell unregistered")
                .slot_base;

            // Bulk-write every registered GPU buffer via the generic
            // type-erased path.  Replaces the old hardcoded downcasts to
            // `SceneBuffer<[f32; 16]>` / `SceneBuffer<InstanceInfo>`.
            let registered_columns: Vec<(ComponentId, ComponentId)> = store.cells[id.0 as usize]
                .as_ref()
                .expect("cell unregistered")
                .dirty_columns_iter()
                .map(|(buffer_id, column_id, _)| (buffer_id, column_id))
                .collect();
            for (buffer_id, column_id) in registered_columns {
                let buffer = store
                    .gpu_buffers
                    .get(&buffer_id)
                    .expect("cell GPU column has no registered buffer");
                if let Some(col_bytes) = cell.column_raw_bytes(column_id) {
                    let row_bytes = buffer.element_size() * rows as usize;
                    if row_bytes > 0 && col_bytes.len() >= row_bytes {
                        buffer.write_rows_raw(&store.queue, &col_bytes[..row_bytes], row_base);
                    }
                }
            }

            // Clear warm-up marks that were just satisfied by the bulk writes.
            let state = store.cells[id.0 as usize]
                .as_mut()
                .expect("cell unregistered");
            for mask in state.dirty_columns.iter_mut().flatten() {
                mask.clear_all();
            }

            let col0 = cell.slot_column();
            {
                let state = store.cells[id.0 as usize]
                    .as_mut()
                    .expect("cell unregistered");
                for row in 0..rows {
                    let local_slot = col0[row as usize];
                    state.slot_scratch[row as usize] = slot_base + local_slot;
                    state.slot_shadow[row as usize] = local_slot;
                }
            }
            store.slot_mirror.write_rows(
                &store.queue,
                &store.cells[id.0 as usize]
                    .as_ref()
                    .expect("cell unregistered")
                    .slot_scratch[..rows as usize],
                row_base,
            );

            ids.push(id);
        }
        (store, ids)
    }

    /// §4.1 eviction (M2b-β, closes the D2 pending-retire disposition and the
    /// region-recycle side of the two §11 carry-forwards): demote a resident
    /// cell out of its region.
    ///
    /// 1. Every retire still queued in `cell`'s pending-retire queue is
    ///    committed **CPU-side immediately** (`cell.commit_retire`: unpin the
    ///    row, bump the registry generation, pool the slot) — **zero VRAM
    ///    writes**. The design originally deferred this commit until the
    ///    region's pin serial completed, to keep a late write from landing in
    ///    a freed (possibly reallocated) region; since this commit never
    ///    touches VRAM there is nothing for a late write to corrupt, so the
    ///    audit remediation (see the M2b-β holistic-audit doc) moves the
    ///    commit earlier, to immediately at eviction. `last_serial` still
    ///    gates the region's byte-range reuse (step 2) — only the retire
    ///    *commit* moved earlier.
    /// 2. Both the row and slot regions are returned to their class's pools
    ///    via `free_pinned(last_serial)` — reusable only once `last_serial`
    ///    completes (`retire_all` drains both pool families every boundary).
    /// 3. The `CellGpuState` (gen-shadow slice, dirty words, slot
    ///    scratch/shadow) is dropped outright — re-created at the region's
    ///    next promotion via `register_cell`'s rebuild + tail scrub.
    ///    CPU-side cell data persists (host memory is authoritative); the
    ///    `cells` slot for `id` becomes `None` and stays `None` for the life
    ///    of the store — `CellId`s are NOT recycled, unlike the row/slot
    ///    regions they used to index.
    ///
    /// Panics if `id` was already unregistered (double-eviction is a caller
    /// bug, not a runtime condition to tolerate).
    pub fn unregister_cell(&mut self, id: CellId, cell: &mut CellStorage, last_serial: u64) {
        let idx = id.0 as usize;
        let state = self.cells[idx].take().expect("cell already unregistered");
        debug_assert!(
            state.pending.back().map_or(true, |q| q.serial <= last_serial),
            "unregister_cell: last_serial must dominate every queued pending serial — the region pin IS the C6 protection for in-flight reads"
        );
        for QueuedRetire { pending, .. } in state.pending {
            cell.commit_retire(pending);
        }
        self.row_pools[state.class].free_pinned(state.row_base, last_serial);
        self.slot_pools[state.class].free_pinned(state.slot_base, last_serial);
        // `state` (gen-shadow, dirty masks, slot scratch/shadow) drops here.
    }

    /// The single mutation path for the GPU-mirrored transform column (§4):
    /// writes the core column AND sets the row's dirty bit in one operation.
    /// False for stale/invalid handles.
    ///
    /// This is the hardcoded equivalent of [`Self::write_gpu`] — `[f32; 16]`
    /// is a bare type, not a derive'd struct, but uses the same generic
    /// `dirty_columns` machinery internally.
    ///
    /// Also stamps the handle's generation into the slot-indexed generation
    /// buffer (adaptation to §5/§7): the design's "written by retirement"
    /// trigger only ever bumps a slot's entry on retire, so a slot's *first*
    /// generation (assigned at `alloc`, which does not pass through
    /// `SceneGpuStore` at all) would otherwise never reach VRAM until that
    /// slot is later retired. The stamp is shadow-gated (`gen_shadow`): a
    /// generation reaches VRAM on the first write after alloc and on
    /// retirement — NOT per `write_transform` call — so repeat writes to a
    /// live handle (the §4 hot path) issue zero generation-buffer traffic
    /// while the buffer still mirrors `HandleRegistry::generations()` for
    /// every allocated slot (C6).
    pub fn write_transform(
        &self,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        m: &[f32; 16],
        _sim: &impl SimulateWitness,
    ) -> bool {
        debug_assert_eq!(
            self.phase,
            Phase::Write,
            "mutation outside the write window"
        );
        let state = self.cells[id.0 as usize]
            .as_ref()
            .expect("cell unregistered");
        let Some(row) = cell.row_of(handle) else {
            return false;
        };
        if cell.is_row_pinned(row) {
            return false; // in-flight retirement: logically deleted (§8) — no further mutation
        }
        let col = cell
            .column_for_mut::<[f32; 16]>()
            .expect("cell has no [f32; 16] transform column");
        col[row as usize] = *m;
        let comp_id = component_id::<[f32; 16]>();
        state
            .dirty_column(comp_id)
            .expect("transform dirty mask missing — register_gpu_buffer not called for [f32; 16]")
            .mark(row);
        self.write_generation(state, handle.index(), handle.generation());
        true
    }

    /// The GPU-mirrored [`InstanceInfo`] column companion to
    /// [`Self::write_transform`] (M3-α T4, cull's token→mesh link): writes
    /// the core column AND sets the row's dirty bit in one operation. False
    /// for stale/invalid handles. Requires the cell to carry an
    /// `InstanceInfo` column (e.g. built via `SpatialCell::with_transform`);
    /// panics otherwise, mirroring `write_transform`'s own unconditional
    /// `.expect()` on its column.
    ///
    /// This is the hardcoded equivalent of [`Self::write_gpu`] — `InstanceInfo`
    /// is a bare type, not a derive'd struct, but uses the same generic
    /// `dirty_columns` machinery internally.
    ///
    /// Body is IDENTICAL to `write_transform`'s minus the generation stamp:
    /// **the generation stamp stays transform-only, by design — one
    /// stamping path.** `write_transform`'s doc explains why the stamp lives
    /// there (a slot's first generation, assigned at `alloc`, must reach VRAM
    /// on *some* write path, and retirement is the other trigger); splitting
    /// it across two mirrored-column writers would mean either double-
    /// stamping (harmless but redundant — the stamp is shadow-gated and
    /// idempotent) or deciding which column "owns" a given slot's first
    /// stamp, a distinction with no purpose. Keeping the stamp solely on
    /// `write_transform` means every handle's first VRAM generation write is
    /// guaranteed by the ONE column every registered cell is required to
    /// carry (transforms, C5-mandatory) rather than the optional one.
    pub fn write_instance_info(
        &self,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        info: InstanceInfo,
        _sim: &impl SimulateWitness,
    ) -> bool {
        debug_assert_eq!(
            self.phase,
            Phase::Write,
            "mutation outside the write window"
        );
        let state = self.cells[id.0 as usize]
            .as_ref()
            .expect("cell unregistered");
        let Some(row) = cell.row_of(handle) else {
            return false;
        };
        if cell.is_row_pinned(row) {
            return false; // in-flight retirement: logically deleted (§8) — no further mutation
        }
        let col = cell
            .column_for_mut::<InstanceInfo>()
            .expect("cell has no InstanceInfo column — register via SpatialCell::with_transform");
        col[row as usize] = info;
        let comp_id = component_id::<InstanceInfo>();
        state
            .dirty_column(comp_id)
            .expect(
                "InstanceInfo dirty mask missing — register_gpu_buffer not called for InstanceInfo",
            )
            .mark(row);
        true
    }

    /// §5 flow step 1: liveness-dead + pinned + enqueued against `serial`.
    /// Registry and GPU buffers unchanged until the serial completes.
    pub fn free_deferred(
        &mut self,
        id: CellId,
        cell: &mut CellStorage,
        handle: Handle,
        serial: u64,
        _sim: &impl SimulateWitness,
    ) -> bool {
        debug_assert_eq!(
            self.phase,
            Phase::Write,
            "free_deferred outside the write window"
        );
        let Some(pending) = cell.mark_pending_retire(handle) else {
            return false;
        };
        let state = self.cells[id.0 as usize]
            .as_mut()
            .expect("cell unregistered");
        debug_assert!(
            state.pending.back().map_or(true, |q| q.serial <= serial),
            "free_deferred serials must be nondecreasing per cell — the retire \
             drain's FIFO early-break would silently stall retirement behind an \
             out-of-order serial"
        );
        state.pending.push_back(QueuedRetire { pending, serial });
        true
    }

    /// §5 flow step 3, frame boundary, runs FIRST: for every cell, drain that
    /// cell's queue (FIFO, early-break on the first incomplete serial)
    /// against `tracker.completed()`; for every drained entry write the new
    /// generation to VRAM, then commit in the registry — the gen bump
    /// reaches the GPU before the slot can be re-allocated (C6). Returns the
    /// total number of slots retired across every cell.
    ///
    /// Also drains both pool families (M2b-β §4.1 eviction): every class's
    /// `row_pools`/`slot_pools` recycles any region pinned by
    /// `unregister_cell` whose serial has now completed, at exactly this
    /// frame boundary — the same watermark this cell-level drain uses.
    pub(crate) fn retire_all(&mut self, cells: &mut [CellSlot<'_>]) -> u32 {
        debug_assert_eq!(
            self.phase,
            Phase::Write,
            "retire_all must open the frame boundary"
        );
        self.phase = Phase::Retired;
        let done = self.tracker.completed();
        let mut drained = 0u32;
        for slot in cells.iter_mut() {
            let idx = slot.id.0 as usize;
            loop {
                let state = self.cells[idx].as_ref().expect("cell unregistered");
                let ready = matches!(state.pending.front(), Some(front) if front.serial <= done);
                if !ready {
                    break; // FIFO serials: everything behind is also incomplete
                }
                let QueuedRetire { pending, .. } = self.cells[idx]
                    .as_mut()
                    .expect("cell unregistered")
                    .pending
                    .pop_front()
                    .unwrap();
                // Retirement always bumps the generation, so the shadow-gated
                // write always lands in VRAM (and updates the shadow) before
                // the registry commit can recycle the slot (C6).
                self.write_generation(
                    self.cells[idx].as_ref().expect("cell unregistered"),
                    pending.slot,
                    pending.next_gen,
                );
                slot.cell.commit_retire(pending);
                drained += 1;
            }
        }
        for pool in self.row_pools.iter_mut().chain(self.slot_pools.iter_mut()) {
            pool.drain_completed(done);
        }
        drained
    }

    /// Frame-boundary compaction (§4): every moved row's destination is
    /// marked dirty in the transform mask (AND the instance-info mask,
    /// M3-α T4 — a moved row carries both mirrored columns to their new
    /// position) so the next sync re-uploads it. The slot mirror is NOT
    /// marked here — `sync_all`'s boundary scan detects moved slots on its
    /// own.
    pub(crate) fn compact_all(&mut self, cells: &mut [CellSlot<'_>]) {
        self.compact_all_gated(cells, |_| true);
    }

    /// Lease-gated variant of [`Self::compact_all`] (§9.2.1, contract #32).
    /// Identical sweep, except `ready` is consulted once per cell (called
    /// with that cell's `CellId`) and a cell it reports NOT ready for is
    /// excluded from THIS boundary's compaction entirely — its holes
    /// persist to the next boundary, the same "deferred" shape
    /// `CellStorage::compact_report` already uses for a pinned tail
    /// (`cell.rs`'s doc). Callers build `ready` from
    /// `gpu::HarvestPipeline::compaction_ready` for each cell's
    /// `(LeaseMask, outstanding leases)` pair — this store has no notion of
    /// leases itself (see that method's ownership-gap doc). `compact_all`
    /// is exactly this with an always-ready gate, so the two can never
    /// silently diverge in behavior.
    pub(crate) fn compact_all_gated(
        &mut self,
        cells: &mut [CellSlot<'_>],
        mut ready: impl FnMut(CellId) -> bool,
    ) {
        debug_assert_eq!(
            self.phase,
            Phase::Retired,
            "compact_all must follow retire_all"
        );
        self.phase = Phase::Compacted;
        for slot in cells.iter_mut() {
            if !ready(slot.id) {
                continue;
            }
            let state = self.cells[slot.id.0 as usize]
                .as_ref()
                .expect("cell unregistered");
            slot.cell.compact_report(|_from, to| {
                // Mark every registered GPU buffer column's dirty mask for
                // the moved row, so the next sync re-uploads everything.
                // The slot mirror is NOT marked here — `sync_all`'s
                // self-healing boundary scan detects moved slots on its own.
                for mask in state.dirty_columns.iter().flatten() {
                    mask.mark(to);
                }
            });
        }
    }

    /// Frame-boundary upload (§4): coalesced dirty-row write of each cell's
    /// GPU-mirrored columns into their disjoint slices of the shared SSBOs,
    /// then the slot mirror via the self-healing boundary scan — every
    /// occupied row whose shadow disagrees with the slot column is marked,
    /// staged, and uploaded, regardless of HOW the slot got there. Closes
    /// the boundary (next phase is the write window). Post-condition: mirror
    /// entries `[row_base, row_base + rows_in_use)` equal
    /// `slot_base + slot_column()[row]` exactly.
    pub(crate) fn sync_all(&mut self, cells: &mut [CellSlot<'_>]) -> SyncStats {
        debug_assert_eq!(
            self.phase,
            Phase::Compacted,
            "sync_all must follow compact_all"
        );
        self.phase = Phase::Write;
        let mut total = SyncStats {
            ranges: 0,
            bytes: 0,
        };
        for slot in cells.iter_mut() {
            let rows = slot.cell.rows_in_use() as usize;
            let cell_ref = &*slot.cell; // reborrow as shared for column reads

            // Sync every dirty GPU-mirrored column.
            let state = self.cells[slot.id.0 as usize]
                .as_ref()
                .expect("cell unregistered");
            for (buffer_id, column_id, dirty_mask) in state.dirty_columns_iter() {
                if let Some(buffer) = self.gpu_buffers.get(&buffer_id) {
                    if let Some(col_bytes) = cell_ref.column_raw_bytes(column_id) {
                        let row_count = rows.min(col_bytes.len() / buffer.element_size());
                        if row_count > 0 {
                            let stats = buffer.sync_region(
                                &self.queue,
                                &col_bytes[..row_count * buffer.element_size()],
                                state.row_base,
                                dirty_mask,
                            );
                            total.ranges += stats.ranges;
                            total.bytes += stats.bytes;
                        }
                    }
                }
            }

            // Self-healing boundary scan — the ONLY slot-mirror dirty
            // trigger (Task 4 re-review): compare every occupied row's
            // shadow against the authoritative slot column and mark exactly
            // the mismatches, whatever moved the slot there (write after
            // alloc, compaction swap, or an alloc re-occupying a vacated row
            // that is never written — the ghost-duplicate case no per-event
            // trigger caught). O(rows) u32 compares per cell per boundary;
            // uploads only actual mismatches.
            let col0 = slot.cell.slot_column();
            let state = self.cells[slot.id.0 as usize]
                .as_mut()
                .expect("cell unregistered");
            for row in 0..rows as u32 {
                let expect = col0[row as usize];
                if state.slot_shadow[row as usize] != expect {
                    // Audit A, G3: mirror `write_generation`'s release-assert
                    // (scene_store.rs, `write_generation`) so a slot value
                    // that is allocated but never written (tombstone-headroom
                    // exhaustion after extreme generation churn — `expect`
                    // beyond `slot_capacity`) fails loud here instead of
                    // silently mirroring this row onto a NEIGHBOR cell's
                    // generation region (fail-open C6). Release, not
                    // debug-only: this is a corruption guard, not a debug
                    // convenience.
                    assert!(
                        expect < state.slot_capacity,
                        "slot {expect} beyond region capacity {} — mirror must never point into a neighbor's region",
                        state.slot_capacity
                    );
                    state.dirty_slots.mark(row);
                    state.slot_scratch[row as usize] = state.slot_base + expect;
                    state.slot_shadow[row as usize] = expect;
                }
            }
            let state = self.cells[slot.id.0 as usize]
                .as_ref()
                .expect("cell unregistered");
            let stats = self.slot_mirror.sync_region(
                &self.queue,
                &state.slot_scratch[..rows],
                state.row_base,
                &state.dirty_slots,
            );
            total.ranges += stats.ranges;
            total.bytes += stats.bytes;
        }
        total
    }

    pub fn row_region_base(&self, id: CellId) -> u32 {
        self.cells[id.0 as usize]
            .as_ref()
            .expect("cell unregistered")
            .row_base
    }

    pub fn transform_buffer(&self) -> &wgpu::Buffer {
        let id = component_id::<[f32; 16]>();
        self.gpu_buffers
            .get(&id)
            .expect("transform buffer not registered")
            .buffer()
    }

    /// Generic counterpart to [`Self::transform_buffer`]/
    /// [`Self::instance_info_buffer`] for a caller-registered type: the same
    /// `ComponentId` lookup those two do by hand for their hardcoded types,
    /// available for any `T` previously passed to
    /// [`Self::register_gpu_buffer`]. `None` if `T` was never registered —
    /// callers that know their type is a required built-in should keep using
    /// the specific accessor (whose `.expect()` gives a clearer panic
    /// message); this is for generic/optional columns.
    pub fn buffer_for<T: HasTypeToken + 'static>(&self) -> Option<&wgpu::Buffer> {
        let id = self
            .gpu_partners
            .resolve(<T as HasTypeToken>::type_token().id());
        self.gpu_buffers.get(&id).map(|b| b.buffer())
    }

    /// `ComponentId`-keyed counterpart to [`Self::buffer_for`], for callers
    /// that only have the id (e.g. from `GpuColumnDesc::field_token.id()`)
    /// and not the concrete registered type — notably, `#[derive(SceneStore)]`
    /// `#[gpu]` fields, whose per-field wrapper type is deliberately
    /// unnameable from outside the macro's own generated code (see
    /// `pulsar_scenedb_derive`'s `FieldInfo::gpu_wrapper` doc). Reading a
    /// derive-mirrored field's buffer back — in a test, in editor tooling,
    /// or in `Self::write_row_bytes`'s own caller, [`crate::gpu::world_mirror`]
    /// — goes through this, not `buffer_for::<Wrapper>()`.
    pub fn buffer_for_id(&self, id: ComponentId) -> Option<&wgpu::Buffer> {
        let id = self.gpu_partners.resolve(id);
        self.gpu_buffers.get(&id).map(|b| b.buffer())
    }

    /// Non-borrowing snapshot of a reflected GPU partner. Growable-buffer
    /// implementations clone the physical handle and capture its allocation
    /// epoch under the same lock, so bind-group caches cannot pair a new
    /// handle with an old epoch. Fixed buffers report epoch `0`.
    pub fn gpu_buffer_snapshot_for_id(
        &self,
        field_id: ComponentId,
    ) -> Option<(wgpu::Buffer, u64, GpuColumnDesc)> {
        let descriptor = self.gpu_partners.descriptor(field_id)?;
        let id = self.gpu_partners.resolve(field_id);
        if let Some(buffer) = self.gpu_buffers.get(&id) {
            return Some((buffer.buffer().clone(), 0, descriptor));
        }
        if let Some(buffer) = self.growable_gpu_buffers.get(&id) {
            let (buffer, epoch) = buffer.buffer_snapshot();
            return Some((buffer, epoch, descriptor));
        }
        if let Some(buffer) = self.dirty_tracked_gpu_buffers.get(&id) {
            let (buffer, epoch) = buffer.buffer_snapshot();
            return Some((buffer, epoch, descriptor));
        }
        if let Some(buffer) = self.once_gpu_buffers.get(&id) {
            let (buffer, epoch) = buffer.buffer_snapshot();
            return Some((buffer, epoch, descriptor));
        }
        None
    }

    /// Stable-key counterpart to [`Self::gpu_buffer_snapshot_for_id`].
    /// This is the preferred renderer seam for
    /// `#[gpu(buffer = "...")]`: the key remains stable even when the first
    /// component-local wrapper chosen as the canonical id changes.
    pub fn gpu_buffer_snapshot_for_key(
        &self,
        key: &str,
    ) -> Option<(wgpu::Buffer, u64, GpuColumnDesc)> {
        let id = self.gpu_partners.id_for_key(key)?;
        self.gpu_buffer_snapshot_for_id(id)
    }

    /// Raw, `ComponentId`-keyed counterpart to [`Self::register_gpu_buffer`]
    /// + [`GpuBufferDispatch::write_rows_raw`], for callers that only have a
    /// `ComponentId` at hand (not the concrete `T`) — e.g. a generic helper
    /// walking `GpuColumnSet::gpu_columns()`'s per-field metadata, which
    /// carries each field's `TypeToken`/`ComponentId` but not its Rust type.
    ///
    /// Bypasses dirty tracking, `CellStorage`, and `Handle` entirely, same as
    /// `write_rows_raw` itself — an unconditional `queue.write_buffer` at
    /// `row`. This is the primitive [`crate::gpu::world_mirror`] builds on to
    /// mirror `World`-owned, cell-free component fields: `row` there is the
    /// stable component-local row resolved by `GpuMirrorHandle`, which needs
    /// no `CellId`/region concept at all.
    ///
    /// Returns `false` (a no-op, not a panic) if `id` was never registered
    /// via `register_gpu_buffer` — a caller inserting a component before its
    /// GPU buffer is wired up is expected during bring-up, not a bug.
    pub fn write_row_bytes(
        &self,
        id: ComponentId,
        queue: &wgpu::Queue,
        data: &[u8],
        row: u32,
    ) -> bool {
        let id = self.gpu_partners.resolve(id);
        match self.gpu_buffers.get(&id) {
            Some(buf) => {
                buf.write_rows_raw(queue, data, row);
                true
            }
            None => false,
        }
    }

    /// Growable counterpart to [`Self::write_row_bytes`], for `id`s
    /// registered via [`Self::register_growable_gpu_buffer`] — grows the
    /// buffer first if `row` doesn't fit the current capacity.
    ///
    /// Returns `None` if `id` was never registered as growable (mirrors
    /// `write_row_bytes`'s "unregistered = no-op" contract exactly — the
    /// caller can't tell "not registered" from "registered fixed instead"
    /// from this alone, which is fine: [`crate::gpu::world_mirror`]'s
    /// dispatch tries `write_row_bytes` first and only falls back to this
    /// method when that returns `false`, so by the time this is called,
    /// "not found here" really does mean "not registered at all").
    ///
    /// Returns `Some(Err(CapacityError))` only if the buffer was registered
    /// with a `max_capacity` ceiling (via `register_growable_gpu_buffer`)
    /// and `row` exceeds it — registrations made with `max_capacity: None`
    /// (the recommendation for World-mirrored columns) can never reach this
    /// case, since unbounded growth cannot fail.
    pub fn write_row_bytes_growing(
        &self,
        id: ComponentId,
        queue: &wgpu::Queue,
        data: &[u8],
        row: u32,
    ) -> Option<Result<(), CapacityError>> {
        let id = self.gpu_partners.resolve(id);
        self.growable_gpu_buffers
            .get(&id)
            .map(|buf| buf.write_row_growing(queue, row, data))
    }

    /// Lock-safe access to a growable column's current buffer, by
    /// `ComponentId` — the growable-buffer counterpart to
    /// [`Self::buffer_for_id`]. See [`GrowableGpuBufferDispatch::with_buffer`]
    /// for why this is callback-shaped rather than returning `&wgpu::Buffer`
    /// directly. Silently does nothing if `id` isn't registered as growable.
    pub fn with_growable_buffer_for_id(&self, id: ComponentId, f: &mut dyn FnMut(&wgpu::Buffer)) {
        let id = self.gpu_partners.resolve(id);
        if let Some(buf) = self.growable_gpu_buffers.get(&id) {
            buf.with_buffer(f);
        }
    }

    /// `epoch()` of a growable column's buffer, by `ComponentId` — bump
    /// count since registration; compare against a previously-observed value
    /// to know whether a bind group built against this column needs
    /// rebuilding. `None` if `id` isn't registered as growable.
    pub fn growable_epoch_for_id(&self, id: ComponentId) -> Option<u64> {
        let id = self.gpu_partners.resolve(id);
        self.growable_gpu_buffers.get(&id).map(|buf| buf.epoch())
    }

    /// Current capacity of a growable column's buffer, by `ComponentId`.
    /// `None` if `id` isn't registered as growable.
    pub fn growable_capacity_for_id(&self, id: ComponentId) -> Option<u32> {
        let id = self.gpu_partners.resolve(id);
        self.growable_gpu_buffers.get(&id).map(|buf| buf.capacity())
    }

    /// Registers a dirty-tracked, growable World-mirrored column — for
    /// `#[gpu(mirror = DirtyTracked)]` fields (the default `#[gpu]` mode).
    /// Writes go through [`Self::mark_gpu_row_dirty`] (CPU-side bookkeeping
    /// only) instead of landing on the GPU immediately; [`Self::flush_gpu_mirror`]
    /// performs the actual, coalesced upload. Like
    /// [`Self::register_growable_gpu_buffer`], never bounded by a
    /// `max_capacity` — a World-mirrored column with a capacity ceiling has
    /// no `Result` to fail through on an ordinary `world.insert()` call.
    pub fn register_dirty_tracked_gpu_buffer<T: Pod + Send + Sync + HasTypeToken + 'static>(
        &mut self,
        initial_capacity: u32,
        device: &Arc<wgpu::Device>,
        label: &str,
    ) {
        let field_id = <T as HasTypeToken>::type_token().id();
        self.gpu_partners
            .claim_residency(field_id, GpuBufferResidency::DirtyTrackedWorld);
        let id = self.gpu_partners.resolve(field_id);
        assert!(
            !self.gpu_buffers.contains_key(&id)
                && !self.growable_gpu_buffers.contains_key(&id)
                && !self.once_gpu_buffers.contains_key(&id),
            "GPU partner {:?} is already registered through a different allocation path",
            id,
        );
        if self.dirty_tracked_gpu_buffers.contains_key(&id) {
            return;
        }
        let buffer = DirtyTrackedSceneBuffer::<T>::new_with_reallocation_policy(
            Arc::clone(device),
            label,
            initial_capacity,
            self.dirty_tracked_reallocation_policy,
        );
        self.dirty_tracked_gpu_buffers.insert(id, Box::new(buffer));
    }

    /// Registers a deferred one-time-handoff World column. Unlike
    /// [`Self::register_dirty_tracked_gpu_buffer`], this keeps no persistent
    /// CPU value shadow: only writes waiting for the next mirror flush are
    /// retained. Removal queues an explicit zero tombstone, and re-insertion
    /// begins a new component-presence lifetime with a new handoff.
    pub fn register_once_gpu_buffer<T: Pod + Send + Sync + HasTypeToken + 'static>(
        &mut self,
        initial_capacity: u32,
        device: &Arc<wgpu::Device>,
        label: &str,
    ) {
        let field_id = <T as HasTypeToken>::type_token().id();
        self.gpu_partners
            .claim_residency(field_id, GpuBufferResidency::OnceWorld);
        let id = self.gpu_partners.resolve(field_id);
        assert!(
            !self.gpu_buffers.contains_key(&id)
                && !self.growable_gpu_buffers.contains_key(&id)
                && !self.dirty_tracked_gpu_buffers.contains_key(&id),
            "GPU partner {:?} is already registered through a different allocation path",
            id,
        );
        if self.once_gpu_buffers.contains_key(&id) {
            return;
        }
        let buffer = OnceSceneBuffer::<T>::new(Arc::clone(device), label, initial_capacity);
        self.once_gpu_buffers.insert(id, Box::new(buffer));
    }

    /// Registers the authoritative per-component presence column used by
    /// the World mirror. `component_id` is the owning component type's id;
    /// it is intentionally distinct from every field/packed-buffer id.
    pub fn register_component_presence_buffer(
        &mut self,
        component_id: ComponentId,
        initial_capacity: u32,
        label: &str,
    ) {
        self.component_presence_buffers.insert(
            component_id,
            ComponentPresenceBuffer::new_with_reallocation_policy(
                Arc::clone(&self.device),
                label,
                initial_capacity,
                self.dirty_tracked_reallocation_policy,
            ),
        );
    }

    /// Marks `row` dirty with `data`'s bytes, for a column registered via
    /// [`Self::register_dirty_tracked_gpu_buffer`] — no GPU work happens
    /// here at all; [`Self::flush_gpu_mirror`] does the actual upload.
    /// Returns `false` if `id` isn't registered dirty-tracked (mirrors
    /// [`Self::write_row_bytes`]'s "unregistered = no-op" contract).
    pub fn mark_gpu_row_dirty(&self, id: ComponentId, row: u32, data: &[u8]) -> bool {
        let id = self.gpu_partners.resolve(id);
        match self.dirty_tracked_gpu_buffers.get(&id) {
            Some(buf) => {
                buf.mark_dirty_bytes(row, data);
                true
            }
            None => false,
        }
    }

    /// Queues one `Once`-mode lifecycle handoff. Returns `false` when `id`
    /// was not registered through [`Self::register_once_gpu_buffer`].
    pub fn queue_gpu_row_once(&self, id: ComponentId, row: u32, data: &[u8]) -> bool {
        let id = self.gpu_partners.resolve(id);
        match self.once_gpu_buffers.get(&id) {
            Some(buf) => {
                buf.queue_handoff_bytes(row, data);
                true
            }
            None => false,
        }
    }

    /// Marks an owning GPU component present. `Some(true)` is the first
    /// handoff of this presence lifetime; `Some(false)` is an ordinary
    /// in-place update; `None` means no presence column was registered.
    pub fn mark_component_present(&self, component_id: ComponentId, row: u32) -> Option<bool> {
        self.component_presence_buffers
            .get(&component_id)
            .map(|buf| buf.mark_present(row))
    }

    /// Writes the explicit absent tombstone (`0`) for an owning component.
    /// Returns whether a present -> absent transition occurred.
    pub fn mark_component_absent(&self, component_id: ComponentId, row: u32) -> Option<bool> {
        self.component_presence_buffers
            .get(&component_id)
            .map(|buf| buf.mark_absent(row))
    }

    /// Uploads dirty-tracked rows, pending `Once` handoffs, and component
    /// presence transitions, coalesced/scattered as appropriate. Call
    /// once per frame — the World-mirror analogue of the cell-mirrored
    /// path's own boundary-phase `sync_all`. A no-op, zero-cost beyond one
    /// empty-map iteration when no World buffers were registered.
    pub fn flush_gpu_mirror(&self, queue: &wgpu::Queue) -> SyncStats {
        let mut total = SyncStats {
            ranges: 0,
            bytes: 0,
        };
        for buf in self.dirty_tracked_gpu_buffers.values() {
            let stats = buf.flush(queue);
            total.ranges += stats.ranges;
            total.bytes += stats.bytes;
        }
        for buf in self.once_gpu_buffers.values() {
            let stats = buf.flush(queue);
            total.ranges += stats.ranges;
            total.bytes += stats.bytes;
        }
        for buf in self.component_presence_buffers.values() {
            let stats = buf.flush(queue);
            total.ranges += stats.ranges;
            total.bytes += stats.bytes;
        }
        total
    }

    /// Lock-safe access to a dirty-tracked column's current buffer, by
    /// `ComponentId` — the dirty-tracked counterpart to
    /// [`Self::with_growable_buffer_for_id`]/[`Self::buffer_for_id`].
    pub fn with_dirty_tracked_buffer_for_id(
        &self,
        id: ComponentId,
        f: &mut dyn FnMut(&wgpu::Buffer),
    ) {
        let id = self.gpu_partners.resolve(id);
        if let Some(buf) = self.dirty_tracked_gpu_buffers.get(&id) {
            buf.with_buffer(f);
        }
    }

    /// `epoch()` of a dirty-tracked column's buffer, by `ComponentId` — the
    /// dirty-tracked counterpart to [`Self::growable_epoch_for_id`]. `None`
    /// if `id` isn't registered dirty-tracked.
    pub fn dirty_tracked_epoch_for_id(&self, id: ComponentId) -> Option<u64> {
        let id = self.gpu_partners.resolve(id);
        self.dirty_tracked_gpu_buffers
            .get(&id)
            .map(|buf| buf.epoch())
    }

    /// Lock-safe access to an explicit `Once`-mode World buffer.
    pub fn with_once_buffer_for_id(&self, id: ComponentId, f: &mut dyn FnMut(&wgpu::Buffer)) {
        let id = self.gpu_partners.resolve(id);
        if let Some(buf) = self.once_gpu_buffers.get(&id) {
            buf.with_buffer(f);
        }
    }

    pub fn once_epoch_for_id(&self, id: ComponentId) -> Option<u64> {
        let id = self.gpu_partners.resolve(id);
        self.once_gpu_buffers.get(&id).map(|buf| buf.epoch())
    }

    pub fn once_capacity_for_id(&self, id: ComponentId) -> Option<u32> {
        let id = self.gpu_partners.resolve(id);
        self.once_gpu_buffers.get(&id).map(|buf| buf.capacity())
    }

    /// Number of handoffs waiting for the next flush. Primarily useful for
    /// telemetry/tests proving that `Once` retains no row-indexed shadow.
    pub fn once_pending_count_for_id(&self, id: ComponentId) -> Option<usize> {
        let id = self.gpu_partners.resolve(id);
        self.once_gpu_buffers
            .get(&id)
            .map(|buf| buf.pending_count())
    }

    /// Lock-safe access to a component type's `u32` presence buffer.
    pub fn with_component_presence_buffer_for_id(
        &self,
        component_id: ComponentId,
        f: &mut dyn FnMut(&wgpu::Buffer),
    ) {
        if let Some(buf) = self.component_presence_buffers.get(&component_id) {
            buf.with_buffer(f);
        }
    }

    /// Atomically snapshots a component-presence buffer and its allocation
    /// epoch. Renderer bind-group caches should prefer this over separately
    /// calling the callback accessor and [`Self::component_presence_epoch_for_id`].
    pub fn component_presence_buffer_snapshot_for_id(
        &self,
        component_id: ComponentId,
    ) -> Option<(wgpu::Buffer, u64)> {
        self.component_presence_buffers
            .get(&component_id)
            .map(ComponentPresenceBuffer::buffer_snapshot)
    }

    pub fn component_presence_epoch_for_id(&self, component_id: ComponentId) -> Option<u64> {
        self.component_presence_buffers
            .get(&component_id)
            .map(ComponentPresenceBuffer::epoch)
    }

    /// Reserves capacity `n` on every World-mirrored buffer registered so
    /// far (growable, dirty-tracked, Once, and presence maps) — call before a
    /// known-size batch spawn (streaming in a sublevel with M known
    /// objects, spawning a wave of enemies from a design-time-known count)
    /// to move every affected buffer's reallocation cost off the per-insert
    /// critical path, in one call. See
    /// [`crate::world::World::reserve_gpu_mirror_capacity`] for the
    /// `World`-level convenience this backs.
    ///
    /// Stops at the first error and returns it — a caller reserving ahead
    /// of a batch wants to know immediately if some column can't grow that
    /// far (e.g. it hit `wgpu::Limits::max_buffer_size`, see
    /// `DynamicGpuBuffer::ensure_capacity`'s doc), not partway through
    /// spawning the batch itself. Buffers already visited before the
    /// failing one keep whatever capacity they were successfully grown to
    /// — this is a best-effort batch operation, not transactional.
    pub fn reserve_world_mirror_capacity(
        &self,
        queue: &wgpu::Queue,
        n: u32,
    ) -> Result<(), CapacityError> {
        for buf in self.growable_gpu_buffers.values() {
            buf.reserve(queue, n)?;
        }
        for buf in self.dirty_tracked_gpu_buffers.values() {
            buf.reserve(queue, n)?;
        }
        for buf in self.once_gpu_buffers.values() {
            buf.reserve(queue, n)?;
        }
        for buf in self.component_presence_buffers.values() {
            buf.reserve(queue, n)?;
        }
        Ok(())
    }

    /// Reserve only the World value columns and presence column owned by one
    /// component type.
    ///
    /// This is the component-local counterpart to
    /// [`Self::reserve_world_mirror_capacity`]. It deliberately does not
    /// reserve the global entity-generation buffer, which belongs to
    /// [`crate::gpu::GenerationMirror`] and remains `Entity::index()`-keyed.
    /// Named partners are resolved to their canonical physical allocation;
    /// World registration guarantees that allocation has exactly one owner.
    pub fn reserve_world_component_capacity(
        &self,
        queue: &wgpu::Queue,
        owner_component_id: ComponentId,
        capacity: u32,
    ) -> Result<(), CapacityError> {
        for (&id, buf) in &self.growable_gpu_buffers {
            if self.gpu_partners.world_owner(id) == Some(owner_component_id) {
                buf.reserve(queue, capacity)?;
            }
        }
        for (&id, buf) in &self.dirty_tracked_gpu_buffers {
            if self.gpu_partners.world_owner(id) == Some(owner_component_id) {
                buf.reserve(queue, capacity)?;
            }
        }
        for (&id, buf) in &self.once_gpu_buffers {
            if self.gpu_partners.world_owner(id) == Some(owner_component_id) {
                buf.reserve(queue, capacity)?;
            }
        }
        if let Some(buf) = self.component_presence_buffers.get(&owner_component_id) {
            buf.reserve(queue, capacity)?;
        }
        Ok(())
    }

    /// Shrinks every World-mirrored buffer registered so far (both maps) to
    /// the smallest capacity that still covers `highest_live_row`, with
    /// `slack_factor` headroom. See
    /// [`crate::gpu::DynamicGpuBuffer::shrink_to_fit`]'s doc — same
    /// semantics, applied across every registered column at once. Call at a
    /// natural boundary (a level unload, a large despawn batch settling),
    /// not every frame; this is a real GPU-to-GPU copy per buffer that
    /// actually shrinks, same cost profile as growth.
    pub fn shrink_world_mirror_to_fit(
        &self,
        queue: &wgpu::Queue,
        highest_live_row: u32,
        slack_factor: f32,
    ) {
        for buf in self.growable_gpu_buffers.values() {
            buf.shrink_to_fit(queue, highest_live_row, slack_factor);
        }
        for buf in self.dirty_tracked_gpu_buffers.values() {
            buf.shrink_to_fit(queue, highest_live_row, slack_factor);
        }
        for buf in self.once_gpu_buffers.values() {
            buf.shrink_to_fit(queue, highest_live_row, slack_factor);
        }
        for buf in self.component_presence_buffers.values() {
            buf.shrink_to_fit(queue, highest_live_row, slack_factor);
        }
    }

    /// Shrink only one World component's value and presence allocations.
    /// `highest_live_row` is in that component's local row domain, not the
    /// global entity-generation domain.
    pub fn shrink_world_component_to_fit(
        &self,
        queue: &wgpu::Queue,
        owner_component_id: ComponentId,
        highest_live_row: u32,
        slack_factor: f32,
    ) -> bool {
        let mut shrank = false;
        for (&id, buf) in &self.growable_gpu_buffers {
            if self.gpu_partners.world_owner(id) == Some(owner_component_id) {
                shrank |= buf.shrink_to_fit(queue, highest_live_row, slack_factor);
            }
        }
        for (&id, buf) in &self.dirty_tracked_gpu_buffers {
            if self.gpu_partners.world_owner(id) == Some(owner_component_id) {
                shrank |= buf.shrink_to_fit(queue, highest_live_row, slack_factor);
            }
        }
        for (&id, buf) in &self.once_gpu_buffers {
            if self.gpu_partners.world_owner(id) == Some(owner_component_id) {
                shrank |= buf.shrink_to_fit(queue, highest_live_row, slack_factor);
            }
        }
        if let Some(buf) = self.component_presence_buffers.get(&owner_component_id) {
            shrank |= buf.shrink_to_fit(queue, highest_live_row, slack_factor);
        }
        shrank
    }

    /// Cull's token→mesh link (M3-α T4, C5 amendment): row-indexed beside
    /// `transform_buffer()`, mirrored via [`Self::write_instance_info`].
    ///
    /// Content contract (same as transforms): a row's bytes are defined only
    /// once it has been written via [`Self::write_instance_info`]. Rows in a
    /// recycled region that were never written may hold a prior tenant's
    /// bytes — readers must not treat `mesh_index` as trusted without the
    /// row having passed a harvest/liveness gate; the M3-β cull shader
    /// additionally bounds-checks `mesh_index` against the mesh table.
    pub fn instance_info_buffer(&self) -> &wgpu::Buffer {
        let id = component_id::<InstanceInfo>();
        self.gpu_buffers
            .get(&id)
            .expect("InstanceInfo buffer not registered")
            .buffer()
    }

    /// Row-indexed global-slot mirror (T4; C6 GPU handle validation).
    ///
    /// Guarantee: after every `sync_all`, entries
    /// `[row_base, row_base + rows_in_use)` of each registered cell mirror
    /// `slot_base + slot_column()[row]` EXACTLY — the boundary scan
    /// self-heals any row whose slot changed by any mechanism (write after
    /// alloc, compaction swap, or an alloc into a previously vacated row
    /// that is never written).
    ///
    /// Mirror entries beyond a cell's `rows_in_use` are stale-but-inert:
    /// compaction shrinks the row count without erasing the mirror tail, so
    /// those entries may hold old slot IDs. Nothing may index the mirror
    /// past the harvested row count (M3 contract) — consumers dispatch over
    /// `rows_in_use`, never region capacity.
    pub fn slot_mirror_buffer(&self) -> &wgpu::Buffer {
        self.slot_mirror.buffer()
    }

    /// Return descriptors for fixed GPU partners the registered cell
    /// actually carries, in its CPU column declaration order.
    ///
    /// World-only growable columns and globally registered fixed buffers
    /// absent from this cell are intentionally excluded. Panics for an
    /// unknown or evicted `CellId`, matching the other cell accessors.
    pub fn gpu_column_descs_for(&self, cell_id: CellId) -> Vec<GpuColumnDesc> {
        self.cells[cell_id.0 as usize]
            .as_ref()
            .expect("cell unregistered")
            .gpu_columns
            .clone()
    }

    pub fn generation_buffer(&self) -> &wgpu::Buffer {
        self.generations.buffer()
    }

    /// Per-cell metadata SSBO (α: allocated, no writer).
    pub fn cell_metadata_buffer(&self) -> &wgpu::Buffer {
        &self.cell_metadata
    }

    // ── Telemetry / snapshot accessors (pub(crate)) ──────────────────────

    /// GPU buffers map for telemetry snapshot.
    #[cfg(feature = "telemetry")]
    pub(crate) fn telemetry_gpu_buffers(
        &self,
    ) -> &HashMap<ComponentId, Box<dyn GpuBufferDispatch>> {
        &self.gpu_buffers
    }

    /// Row region pools for telemetry snapshot.
    #[cfg(feature = "telemetry")]
    pub(crate) fn telemetry_row_pools(&self) -> &[RegionPool] {
        &self.row_pools
    }

    /// Slot region pools for telemetry snapshot.
    #[cfg(feature = "telemetry")]
    pub(crate) fn telemetry_slot_pools(&self) -> &[RegionPool] {
        &self.slot_pools
    }

    /// Per-cell GPU state for telemetry snapshot.
    #[cfg(feature = "telemetry")]
    pub(crate) fn telemetry_cells(&self) -> &[Option<CellGpuState>] {
        &self.cells
    }
}

#[cfg(test)]
mod partner_registry_tests {
    use super::*;
    use crate::token::TypeToken;

    #[repr(transparent)]
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    struct FieldA(f32);
    unsafe impl Pod for FieldA {}

    #[repr(transparent)]
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    struct FieldB(f32);
    unsafe impl Pod for FieldB {}

    #[repr(transparent)]
    #[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
    struct FieldC(u64);
    unsafe impl Pod for FieldC {}

    fn desc(
        field_token: crate::token::TypeToken,
        value_token: crate::token::TypeToken,
        mode: MirrorMode,
        key: Option<&'static str>,
    ) -> GpuColumnDesc {
        GpuColumnDesc {
            field_token,
            value_token,
            field_offset: 0,
            mode,
            buffer_name: "value",
            buffer_key: key,
        }
    }

    #[test]
    fn compatible_named_fields_share_one_canonical_buffer_id() {
        let a = desc(
            TypeToken::of::<FieldA>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            Some("scene.transforms"),
        );
        let b = desc(
            TypeToken::of::<FieldB>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            Some("scene.transforms"),
        );
        let mut registry = GpuPartnerRegistry::default();
        registry.register(a);
        registry.register(b);

        assert_eq!(registry.resolve(a.field_token.id()), a.field_token.id());
        assert_eq!(registry.resolve(b.field_token.id()), a.field_token.id());
        assert_eq!(
            registry.id_for_key("scene.transforms"),
            Some(a.field_token.id())
        );
        assert_eq!(registry.descriptor(b.field_token.id()), Some(b));
    }

    #[test]
    fn equal_display_names_without_explicit_keys_remain_distinct() {
        let a = desc(
            TypeToken::of::<FieldA>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            None,
        );
        let b = desc(
            TypeToken::of::<FieldB>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            None,
        );
        let mut registry = GpuPartnerRegistry::default();
        registry.register(a);
        registry.register(b);

        assert_ne!(
            registry.resolve(a.field_token.id()),
            registry.resolve(b.field_token.id())
        );
    }

    #[test]
    #[should_panic(expected = "incompatible #[gpu(buffer = \"scene.shared\")] row types")]
    fn named_fields_reject_different_row_types_even_when_size_matches() {
        let mut registry = GpuPartnerRegistry::default();
        registry.register(desc(
            TypeToken::of::<FieldA>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            Some("scene.shared"),
        ));
        registry.register(desc(
            TypeToken::of::<FieldB>(),
            TypeToken::of::<u32>(),
            MirrorMode::DirtyTracked,
            Some("scene.shared"),
        ));
    }

    #[test]
    #[should_panic(expected = "incompatible #[gpu(buffer = \"scene.shared\")] row layout/stride")]
    fn named_fields_reject_different_row_layouts() {
        let mut registry = GpuPartnerRegistry::default();
        registry.register(desc(
            TypeToken::of::<FieldA>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            Some("scene.shared"),
        ));
        registry.register(desc(
            TypeToken::of::<FieldC>(),
            TypeToken::of::<u64>(),
            MirrorMode::DirtyTracked,
            Some("scene.shared"),
        ));
    }

    #[test]
    #[should_panic(expected = "incompatible #[gpu(buffer = \"scene.shared\")] mirror modes")]
    fn named_fields_reject_different_mirror_modes() {
        let mut registry = GpuPartnerRegistry::default();
        registry.register(desc(
            TypeToken::of::<FieldA>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            Some("scene.shared"),
        ));
        registry.register(desc(
            TypeToken::of::<FieldB>(),
            TypeToken::of::<f32>(),
            MirrorMode::Once,
            Some("scene.shared"),
        ));
    }

    #[test]
    #[should_panic(expected = "allocation domains")]
    fn named_partner_rejects_fixed_and_world_residency_in_one_store() {
        let a = desc(
            TypeToken::of::<FieldA>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            Some("scene.shared"),
        );
        let b = desc(
            TypeToken::of::<FieldB>(),
            TypeToken::of::<f32>(),
            MirrorMode::DirtyTracked,
            Some("scene.shared"),
        );
        let mut registry = GpuPartnerRegistry::default();
        registry.register(a);
        registry.register(b);
        registry.claim_residency(a.field_token.id(), GpuBufferResidency::FixedCell);
        registry.claim_residency(b.field_token.id(), GpuBufferResidency::DirtyTrackedWorld);
    }
}
