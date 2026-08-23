//! SceneDB is an in-memory scene database with two complementary CPU storage
//! models:
//!
//! - [`World`] is the archetype/component authority used for general authored
//!   scene data, typed queries, relations, schedules, and registered subsystems.
//! - [`CellStorage`] and [`SpatialCell`] provide fixed paged SoA storage for
//!   streaming, SIMD spatial queries, and handle-addressed cell workloads.
//!
//! [`Entity`] identities belong to `World`; [`Handle`] identities belong to the
//! paged cell path. Neither physical archetype rows nor GPU partner rows are
//! derived from an `Entity` index.
//!
//! With the `gpu` feature, `SceneGpuStore` supports both fixed cell mirrors and
//! growable `World` component partners. A `#[gpu]` field defaults to
//! `DirtyTracked`: the CPU component remains canonical and changed
//! component-local rows are uploaded on flush. `#[gpu(mirror = Once)]` is an
//! explicit one-time handoff per component-presence lifetime and retains only
//! transient pending values, not a capacity-sized CPU shadow. Named buffer
//! identities, reflection descriptors, presence rows, and the separate
//! entity-generation buffer let renderers bind these publications without
//! taking scene-data ownership.
//!
//! The crate also provides streaming/phase primitives, asset-oriented GPU
//! subsystems, and experimental local replication building blocks. The core
//! remains graphics-free when built without the `gpu` feature.

pub mod actor;
pub mod archetype;
pub mod cell;
pub mod cell_type;
pub mod component;
pub mod component_store;
pub mod entity;
pub mod handle;
pub mod lease;
pub mod liveness;
pub mod page;
pub mod query;
pub mod registry;
pub mod relation;
pub mod replication;

pub mod schedule;
pub mod simd;
pub mod snapshot;
pub mod spatial;
pub mod time;
pub mod token;
pub mod world;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod subsystem;

#[cfg(feature = "gpu")]
pub mod scene_db;

#[cfg(feature = "telemetry")]
pub mod telemetry;

pub use actor::{Actor, ActorRegistry};
pub use archetype::{Archetype, ArchetypeId, ArchetypeKey};
pub use cell::CellStorage;
pub use cell_type::{CellType, CellTypeError, RegisteredCellType, SceneColumnSet};
pub use component::{component_id, Component, ComponentId};
pub use component_store::{
    __bp_clear_comp_ctx, __bp_set_comp_ctx, __bp_with_comp, __bp_with_comp_ctx, ComponentStore,
};
pub use entity::Entity;
#[cfg(feature = "gpu")]
pub use gpu::{
    gpu_column_descs_for_component, CapacityError, DescriptorsFn, DirtyTrackedGpuBufferDispatch,
    DirtyTrackedReallocationPolicy, DirtyTrackedSceneBuffer, DynamicGpuBuffer, GenerationMirror,
    GpuBufferDispatch, GpuColumnDesc, GpuColumnSet, GpuMirrorHandle, GpuMirrorRegistration,
    GrowableGpuBufferDispatch, GrowableGpuColumnSet, GrowableSceneBuffer, MirrorMode,
};
// Re-exported so `#[derive(SceneStore)]`'s generated code (which lands in
// whatever crate uses the derive, not here) can reach `inventory::submit!`
// via `::pulsar_scenedb::pulsar_reflection::inventory` without that crate
// needing its own direct `pulsar_reflection` dependency. The `wgpu`
// re-export below serves the same purpose for generated registration
// signatures.
pub use handle::Handle;
pub use lease::{Lease, LeaseMask, Scratchpad, DECAY_FRAMES, LEASE_SLOTS};
pub use liveness::LivenessMask;
pub use page::{
    Column, ColumnDesc, GenericColumn, LayoutError, Page, PageLayout, Pod, PodColumn,
    DEFAULT_PAGE_CAPACITY, MAX_PAGE_CAPACITY, MAX_STRIDE_BYTES,
};
// Generated GPU row wrappers use bytemuck's stronger no-uninitialized-byte
// Pod contract. Re-export it so downstream macro expansions do not need a
// direct dependency merely to name the derive traits.
#[doc(hidden)]
pub use bytemuck;
// `SceneStore` expansions live in downstream crates. Route their GPU type
// paths through the same dependency instance SceneDB was compiled with so an
// extension crate does not need a second direct/version-sensitive `wgpu`
// dependency merely for generated registration signatures.
#[cfg(feature = "gpu")]
#[doc(hidden)]
pub use wgpu;
pub use pulsar_reflection;
pub use query::{QueryIter, QueryIterMut, WorldQuery, WorldQueryMut};
pub use registry::{HandleRegistry, NULL_ROW};
pub use relation::{ConflictEntry, ConflictReason, RelationIndex, RelationView};
pub use replication::{
    condition_passes, decode_archetype_key, encode_archetype_key, encode_field_value,
    encode_pod_raw, AuthorityTable, CellRowSnapshot, ChangeTracker, ClientId, ClientInput,
    ComponentDelta, CpuSimulateWitness, Delta, DeltaCompressor, DeltaView, EntityCellMap,
    EntitySnapshot, ErrorCode, EventBatch, EventChannel, FieldDescriptor, Ownership, Reconciler,
    RelevanceSet, Replicable, ReplicatedEvent, ReplicationCondition, ReplicationEncoding,
    ReplicationRegistry, ReplicationSchema, SchemaBuilder, Snapshot,
};
#[cfg(feature = "gpu")]
pub use scene_db::SceneDb;
pub use schedule::Schedule;
pub use snapshot::{LivenessSnapshot, RevocationFlag};
pub use spatial::{
    Aabb, Frustum, InstanceInfo, SpatialCell, INSTANCE_INFO_COLUMN, SPATIAL_COLUMNS,
    TRANSFORM_COLUMN,
};
#[cfg(feature = "gpu")]
pub use subsystem::{Subsystem, SubsystemRegistry};
pub use time::GameTime;
pub use token::{HasTypeToken, TypeToken};
pub use world::World;

#[cfg(feature = "telemetry")]
pub use telemetry::{TelemetryServer, TelemetrySnapshot};
