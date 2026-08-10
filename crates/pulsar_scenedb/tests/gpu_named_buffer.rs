//! Compile-time and metadata coverage for explicit GPU buffer identities.
//!
//! These types intentionally reuse `general_mesh_buf`: macro-generated
//! field wrappers remain distinct CPU-column identities, while the emitted
//! raw value token + explicit key give `SceneGpuStore` enough static
//! information to validate and alias the compatible GPU destination.

use pulsar_scenedb::gpu::{EngineGpuContext, RegionClassConfig, SceneGpuConfig, SceneGpuStore};
use pulsar_scenedb::{
    component_id, gpu_column_descs_for_component, CellStorage, GpuColumnSet, MirrorMode,
    SceneColumnSet, TypeToken,
};
use pulsar_scenedb_derive::SceneStore;

#[derive(SceneStore, Clone, Copy)]
struct MeshReference {
    #[gpu(buffer = "general_mesh_buf")]
    mesh_id: u32,
    #[gpu(mirror = Once, buffer = "general_material_buf")]
    material_id: u32,
    #[gpu]
    local_flags: u32,
}

#[derive(SceneStore, Clone, Copy)]
struct CompatibleMeshReference {
    #[gpu(buffer = "general_mesh_buf", mirror = DirtyTracked)]
    mesh: u32,
}

#[test]
fn derive_emits_static_named_identity_without_changing_display_names() {
    let columns = MeshReference::gpu_columns();
    assert_eq!(columns.len(), 3);

    assert_eq!(columns[0].buffer_name, "mesh_id");
    assert_eq!(columns[0].buffer_key, Some("general_mesh_buf"));
    assert_eq!(columns[0].value_token, TypeToken::of::<u32>());
    assert_eq!(columns[0].mode, MirrorMode::DirtyTracked);

    assert_eq!(columns[1].buffer_name, "material_id");
    assert_eq!(columns[1].buffer_key, Some("general_material_buf"));
    assert_eq!(columns[1].value_token, TypeToken::of::<u32>());
    assert_eq!(columns[1].mode, MirrorMode::Once);

    assert_eq!(columns[2].buffer_name, "local_flags");
    assert_eq!(columns[2].buffer_key, None);
    assert_eq!(columns[2].value_token, TypeToken::of::<u32>());

    assert_ne!(
        columns[0].field_token, columns[0].value_token,
        "the generated wrapper remains the private CellStorage column identity"
    );
}

#[test]
fn compatible_types_can_compile_with_the_same_explicit_key() {
    let first = MeshReference::gpu_columns();
    let second = CompatibleMeshReference::gpu_columns();

    assert_eq!(first[0].buffer_key, second[0].buffer_key);
    assert_eq!(first[0].value_token, second[0].value_token);
    assert_eq!(first[0].mode, second[0].mode);
    assert_ne!(
        first[0].field_token, second[0].field_token,
        "component-local column identities must remain distinct until store registration"
    );
}

#[test]
fn inventory_exposes_the_same_descriptors_for_world_reflection() {
    let descriptors = gpu_column_descs_for_component(component_id::<MeshReference>())
        .expect("SceneStore derive must submit World GPU reflection metadata");
    assert_eq!(descriptors.len(), 3);
    assert_eq!(descriptors[0].buffer_key, Some("general_mesh_buf"));
    assert_eq!(descriptors[1].mode, MirrorMode::Once);
    assert_eq!(descriptors[2].buffer_key, None);
}

fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("scenedb-named-gpu-partner-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

#[test]
fn compatible_named_cell_columns_reuse_one_physical_buffer_and_reflect_per_cell() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(
        &ctx,
        SceneGpuConfig {
            classes: vec![RegionClassConfig {
                capacity: 64,
                max_resident_cells: 2,
            }],
            tombstone_headroom: 8,
            max_cells_metadata: 4,
        },
    );
    const TOTAL_ROWS: u32 = 128;
    MeshReference::register_gpu_columns(&mut store, TOTAL_ROWS, ctx.device());
    CompatibleMeshReference::register_gpu_columns(&mut store, TOTAL_ROWS, ctx.device());

    let first = MeshReference::gpu_columns();
    let second = CompatibleMeshReference::gpu_columns();
    let canonical = store
        .gpu_buffer_id_for_key("general_mesh_buf")
        .expect("named key registered");
    assert_eq!(canonical, first[0].field_token.id());
    assert_eq!(
        store.resolve_gpu_buffer_id(second[0].field_token.id()),
        canonical
    );

    let (named_buffer, named_epoch, named_desc) = store
        .gpu_buffer_snapshot_for_key("general_mesh_buf")
        .expect("named buffer snapshot");
    let (alias_buffer, alias_epoch, alias_desc) = store
        .gpu_buffer_snapshot_for_id(second[0].field_token.id())
        .expect("alias buffer snapshot");
    assert_eq!(
        named_buffer, alias_buffer,
        "aliases must clone the same wgpu allocation"
    );
    assert_eq!(
        (named_epoch, alias_epoch),
        (0, 0),
        "fixed buffers never reallocate"
    );
    assert_eq!(named_desc, first[0]);
    assert_eq!(alias_desc, second[0]);

    let first_cell = CellStorage::from_cell_type(&MeshReference::cell_type(), 64).unwrap();
    let second_cell =
        CellStorage::from_cell_type(&CompatibleMeshReference::cell_type(), 64).unwrap();
    let first_id = store.register_cell(&first_cell, 0).unwrap();
    let second_id = store.register_cell(&second_cell, 0).unwrap();
    assert_eq!(store.gpu_column_descs_for(first_id), first);
    assert_eq!(store.gpu_column_descs_for(second_id), second);
    assert_ne!(
        store.row_region_base(first_id),
        store.row_region_base(second_id)
    );
}

#[test]
#[should_panic(expected = "World GPU buffer")]
fn named_world_buffer_rejects_a_second_component_owner() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(
        &ctx,
        SceneGpuConfig {
            classes: vec![RegionClassConfig {
                capacity: 64,
                max_resident_cells: 1,
            }],
            tombstone_headroom: 8,
            max_cells_metadata: 4,
        },
    );

    MeshReference::register_gpu_columns_growable(&mut store, 8, ctx.device());
    // Both components name a compatible u32 DirtyTracked destination
    // `general_mesh_buf`. That is safe for fixed CellStorage because each
    // cell owns a disjoint row region (proven above), but not for World: on
    // an entity carrying both components, mutation order would otherwise be
    // silent last-write-wins into one physical row.
    CompatibleMeshReference::register_gpu_columns_growable(&mut store, 8, ctx.device());
}
