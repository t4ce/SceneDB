use proc_macro::TokenStream;
use syn::{parse_macro_input, DeriveInput, ItemImpl};

mod cell;
mod gpu;
mod replicate;
mod scene_store;
mod subsystem;

/// Derive `HasTypeToken`, `Pod`, `SceneColumnSet`, and `GpuColumnSet` for a
/// SceneDB component struct.
///
/// # Attributes
///
/// - `#[gpu]` — mark a field as GPU-mirrored (requires the `gpu` feature on
///   `pulsar_scenedb`).
/// - `#[gpu(mirror = Once)]` — one deferred GPU handoff per World component-
///   presence lifetime; removal tombstones it and re-insertion hands off again.
/// - `#[gpu(mirror = DirtyTracked)]` — GPU-mirrored field synced every frame
///   (default for bare `#[gpu]`).
/// - `#[gpu(buffer = "general_mesh_buf")]` — assign a stable, explicit GPU
///   buffer identity. Fixed CellStorage fields may reuse the same key when
///   their row types/layouts and mirror modes match because cells occupy
///   disjoint row regions. Growable World registration permits only one
///   owning component per canonical buffer: two component types would write
///   the same entity row and are rejected instead of becoming
///   last-write-wins.
/// - Options compose, for example
///   `#[gpu(mirror = Once, buffer = "general_mesh_buf")]`.
///
/// GPU fields additionally implement bytemuck's `Pod + Zeroable` contract.
/// This rejects implicit padding because World differential dispatch compares
/// exact shader-row bytes. Packed rows must have no implicit internal or
/// trailing padding; model required shader padding as explicit `#[gpu]`
/// fields.
///
/// Generic structs remain supported when all fields are CPU-only (subject to
/// the usual `Pod + 'static` bounds for stored field types). A generic struct
/// with any `#[gpu]` field is rejected until SceneDB has an explicit
/// per-monomorph GPU inventory/partner registration API; use a concrete
/// wrapper type for GPU-mirrored data today.
#[proc_macro_derive(SceneStore, attributes(gpu))]
pub fn derive_scene_store(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    scene_store::expand(input)
        .unwrap_or_else(|err| err.to_compile_error().into())
        .into()
}

/// Derive a `register_replication` method that builds a
/// [`ReplicationSchema`](pulsar_scenedb::ReplicationSchema) from
/// `#[replicate(encoding = ..., condition = ...)]` field attributes.
///
/// Each annotated field is registered with a real accessor into that named
/// field (not the whole struct), so replicated updates are byte-accurate
/// per field — the field's own type must implement
/// [`Replicable`](pulsar_scenedb::Replicable) (every `Pod` type already
/// does; `String`/`Vec<T>`/`Option<T>` work out of the box too). The struct
/// itself must also implement `Default` — used to fill a placeholder row
/// when an entity is spawned before its real field values arrive.
///
/// # Example
///
/// ```ignore
/// #[derive(Replicate, Default)]
/// struct Health {
///     #[replicate(encoding = DeltaCompressed, condition = SimulatedOnly)]
///     value: f32,
/// }
///
/// let mut registry = ReplicationRegistry::new();
/// Health::register_replication(&mut registry);
/// ```
#[proc_macro_derive(Replicate, attributes(replicate))]
pub fn derive_replicate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    replicate::expand(input)
        .unwrap_or_else(|err| err.to_compile_error().into())
        .into()
}

/// Mark an `impl Foo { .. }` block as a SceneDB subsystem, registering every
/// `#[subsystem_method]`-marked method into Pulsar's reflection database
/// (`pulsar_reflection::DYN_METHOD_REGISTRY`) at link time via
/// `inventory::submit!`.
///
/// `name = "..."` sets the registry key subsystems are dispatched under
/// (via [`pulsar_scenedb::SubsystemRegistry::dispatch`] /
/// [`pulsar_scenedb::SceneDb::dispatch`]); defaults to the type's name.
///
/// # Example
///
/// ```ignore
/// #[scenedb_subsystem(name = "physics")]
/// impl PhysicsSubsystem {
///     #[subsystem_method]
///     pub fn apply_impulse(&mut self, entity: Handle, impulse: [f32; 3]) {
///         // ...
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn scenedb_subsystem(attr: TokenStream, item: TokenStream) -> TokenStream {
    let impl_block = parse_macro_input!(item as ItemImpl);
    subsystem::expand(attr.into(), impl_block)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

/// Marks a method inside a `#[scenedb_subsystem]` impl block as
/// dynamically dispatchable by name. A real (identity) attribute macro —
/// not a bare marker `#[scenedb_subsystem]` has to strip — so it's valid
/// syntax on its own even before `#[scenedb_subsystem]` processes it.
///
/// `#[subsystem_method(category = "...")]` optionally sets a display
/// category, mirroring `engine_class_derive`'s `#[method(...)]`.
#[proc_macro_attribute]
pub fn subsystem_method(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
