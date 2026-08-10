//! GPU lifecycle contract for World-mirrored components.
//!
//! Entity generation and component presence are intentionally orthogonal:
//! removing one component keeps the entity alive and therefore cannot bump
//! its generation. Each GPU-bearing component has an explicit `u32`
//! presence column, while removal also zeroes its value partners as hygiene.

use pulsar_scenedb::component::component_id;
use pulsar_scenedb::gpu::{
    EngineGpuContext, GpuColumnSet, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig,
    SceneGpuStore,
};
use pulsar_scenedb::World;
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;

fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter -- GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("scenedb-world-gpu-mirror-removal-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn scene_cfg() -> SceneGpuConfig {
    SceneGpuConfig {
        classes: vec![RegionClassConfig {
            capacity: 64,
            max_resident_cells: 1,
        }],
        tombstone_headroom: 8,
        max_cells_metadata: 16,
    }
}

fn readback(ctx: &EngineGpuContext, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("world-removal-readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
    ctx.queue().submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("map"));
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let bytes = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    bytes
}

fn read_u32(ctx: &EngineGpuContext, buffer: &wgpu::Buffer, row: u32) -> u32 {
    u32::from_ne_bytes(readback(ctx, buffer, row as u64 * 4, 4).try_into().unwrap())
}

fn read_partner_u32(
    ctx: &EngineGpuContext,
    store: &SceneGpuStore,
    field_id: pulsar_scenedb::ComponentId,
    row: u32,
) -> u32 {
    let (buffer, _epoch, _descriptor) = store
        .gpu_buffer_snapshot_for_id(field_id)
        .expect("registered GPU partner");
    read_u32(ctx, &buffer, row)
}

fn read_presence_u32(
    ctx: &EngineGpuContext,
    store: &SceneGpuStore,
    owner: pulsar_scenedb::ComponentId,
    row: u32,
) -> u32 {
    let (buffer, _epoch) = store
        .component_presence_buffer_snapshot_for_id(owner)
        .expect("registered component-presence buffer");
    read_u32(ctx, &buffer, row)
}

#[derive(SceneStore, Clone, Copy)]
#[repr(C)]
struct LifecycleComponent {
    #[gpu(mirror = Once)]
    mesh: u32,
    #[gpu]
    material: u32,
}

#[test]
fn remove_reinsert_and_despawn_have_distinct_presence_and_generation_semantics() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    LifecycleComponent::register_gpu_columns_growable(&mut store, 2, ctx.device());
    let store = Arc::new(store);
    let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue()));

    let mut world = World::new();
    world.attach_gpu_mirror(mirror.clone());

    // Start with a recycled, nonzero generation so a default-zero GPU row
    // cannot make a false-positive generation assertion pass.
    let old = world.spawn();
    assert!(world.despawn(old));
    let entity = world.spawn();
    assert_ne!(entity.generation(), 0);
    let entity_row = entity.index();

    let columns = LifecycleComponent::gpu_columns();
    let mesh_id = columns
        .iter()
        .find(|desc| desc.buffer_name == "mesh")
        .unwrap()
        .field_token
        .id();
    let material_id = columns
        .iter()
        .find(|desc| desc.buffer_name == "material")
        .unwrap()
        .field_token
        .id();
    let owner = component_id::<LifecycleComponent>();

    world.insert(
        entity,
        LifecycleComponent {
            mesh: 11,
            material: 101,
        },
    );
    let row = world
        .gpu_row::<LifecycleComponent>(entity)
        .expect("component GPU row");
    assert_eq!(store.once_pending_count_for_id(mesh_id), Some(1));
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(store.once_pending_count_for_id(mesh_id), Some(0));
    assert_eq!(read_partner_u32(&ctx, &store, mesh_id, row), 11);
    assert_eq!(read_partner_u32(&ctx, &store, material_id, row), 101);
    assert_eq!(read_presence_u32(&ctx, &store, owner, row), 1);
    let (generation_buffer, _) = mirror.generations().buffer_snapshot();
    assert_eq!(
        read_u32(&ctx, &generation_buffer, entity_row),
        entity.generation()
    );

    // Ordinary updates re-sync DirtyTracked but do not revisit Once.
    world.insert(
        entity,
        LifecycleComponent {
            mesh: 22,
            material: 202,
        },
    );
    assert_eq!(store.once_pending_count_for_id(mesh_id), Some(0));
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(read_partner_u32(&ctx, &store, mesh_id, row), 11);
    assert_eq!(read_partner_u32(&ctx, &store, material_id, row), 202);

    // Removing only this component leaves the entity generation untouched,
    // writes presence=0, and clears both value partners.
    assert!(world.remove::<LifecycleComponent>(entity).is_some());
    assert!(world.is_alive(entity));
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(read_partner_u32(&ctx, &store, mesh_id, row), 0);
    assert_eq!(read_partner_u32(&ctx, &store, material_id, row), 0);
    assert_eq!(read_presence_u32(&ctx, &store, owner, row), 0);
    let (generation_buffer, _) = mirror.generations().buffer_snapshot();
    assert_eq!(
        read_u32(&ctx, &generation_buffer, entity_row),
        entity.generation()
    );

    // Re-insertion starts a new presence lifetime, so Once hands off again.
    world.insert(
        entity,
        LifecycleComponent {
            mesh: 33,
            material: 303,
        },
    );
    assert_eq!(store.once_pending_count_for_id(mesh_id), Some(1));
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(read_partner_u32(&ctx, &store, mesh_id, row), 33);
    assert_eq!(read_partner_u32(&ctx, &store, material_id, row), 303);
    assert_eq!(read_presence_u32(&ctx, &store, owner, row), 1);

    // Multiple lifecycle events in one frame collapse in order: the final
    // re-insert wins over the preceding removal tombstone.
    assert!(world.remove::<LifecycleComponent>(entity).is_some());
    world.insert(
        entity,
        LifecycleComponent {
            mesh: 44,
            material: 404,
        },
    );
    assert_eq!(store.once_pending_count_for_id(mesh_id), Some(2));
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(store.once_pending_count_for_id(mesh_id), Some(0));
    assert_eq!(read_partner_u32(&ctx, &store, mesh_id, row), 44);
    assert_eq!(read_partner_u32(&ctx, &store, material_id, row), 404);
    assert_eq!(read_presence_u32(&ctx, &store, owner, row), 1);

    // Despawn clears component presence/value and independently advances
    // entity generation, protecting both dimensions of stale access.
    assert!(world.despawn(entity));
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(read_partner_u32(&ctx, &store, mesh_id, row), 0);
    assert_eq!(read_partner_u32(&ctx, &store, material_id, row), 0);
    assert_eq!(read_presence_u32(&ctx, &store, owner, row), 0);
    let (generation_buffer, _) = mirror.generations().buffer_snapshot();
    assert_ne!(
        read_u32(&ctx, &generation_buffer, entity_row),
        entity.generation()
    );
}

#[test]
fn late_mirror_attachment_counts_as_the_first_once_handoff() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    LifecycleComponent::register_gpu_columns_growable(&mut store, 2, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    let entity = world.spawn();
    world.insert(
        entity,
        LifecycleComponent {
            mesh: 1,
            material: 2,
        },
    );

    let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue()));
    world.attach_gpu_mirror(mirror.clone());
    // Structurally this is an in-place update, but it is the first handoff to
    // this mirror and must therefore upload Once as well as DirtyTracked.
    world.insert(
        entity,
        LifecycleComponent {
            mesh: 77,
            material: 88,
        },
    );
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    let columns = LifecycleComponent::gpu_columns();
    let mesh_id = columns
        .iter()
        .find(|desc| desc.buffer_name == "mesh")
        .unwrap()
        .field_token
        .id();
    let gpu_row = world
        .gpu_row::<LifecycleComponent>(entity)
        .expect("component GPU row");
    assert_eq!(read_partner_u32(&ctx, &store, mesh_id, gpu_row), 77);
    assert_eq!(
        read_presence_u32(
            &ctx,
            &store,
            component_id::<LifecycleComponent>(),
            gpu_row,
        ),
        1,
    );
    let (generation_buffer, _) = mirror.generations().buffer_snapshot();
    assert_eq!(
        read_u32(&ctx, &generation_buffer, entity.index()),
        entity.generation(),
    );
}

#[derive(SceneStore, Clone, Copy)]
#[repr(C)]
struct FixedWorldComponent {
    #[gpu]
    value: u32,
}

#[test]
#[should_panic(expected = "register_gpu_columns_growable")]
fn fixed_cell_registration_fails_loudly_when_used_as_a_world_mirror() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    // Fixed registration is deliberately the CellStorage path and does not
    // charge Cell-only users for a World component-presence buffer.
    FixedWorldComponent::register_gpu_columns(&mut store, 8, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));
    let entity = world.spawn();
    world.insert(entity, FixedWorldComponent { value: 1 });
}

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed)]
#[repr(C)]
struct PackedOnceComponent {
    #[gpu(mirror = Once)]
    mesh: u32,
    #[gpu(mirror = Once)]
    flags: u32,
}

#[test]
fn packed_once_rows_are_zeroed_and_rehanded_off_after_removal() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    PackedOnceComponent::register_gpu_columns_growable(&mut store, 2, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));
    let entity = world.spawn();
    let packed_id = PackedOnceComponent::packed_gpu_component_id();

    world.insert(entity, PackedOnceComponent { mesh: 9, flags: 10 });
    let gpu_row = world
        .gpu_row::<PackedOnceComponent>(entity)
        .expect("component GPU row");
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let (buffer, _, _) = store
        .gpu_buffer_snapshot_for_id(packed_id)
        .expect("packed Once buffer");
    assert_eq!(
        readback(&ctx, &buffer, gpu_row as u64 * 8, 8),
        [9u32.to_ne_bytes(), 10u32.to_ne_bytes()].concat(),
    );

    assert!(world.remove::<PackedOnceComponent>(entity).is_some());
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let (buffer, _, _) = store
        .gpu_buffer_snapshot_for_id(packed_id)
        .expect("packed Once buffer");
    assert_eq!(
        readback(&ctx, &buffer, gpu_row as u64 * 8, 8),
        vec![0; 8]
    );

    world.insert(
        entity,
        PackedOnceComponent {
            mesh: 19,
            flags: 20,
        },
    );
    assert_eq!(world.gpu_row::<PackedOnceComponent>(entity), Some(gpu_row));
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    let (buffer, _, _) = store
        .gpu_buffer_snapshot_for_id(packed_id)
        .expect("packed Once buffer");
    assert_eq!(
        readback(&ctx, &buffer, gpu_row as u64 * 8, 8),
        [19u32.to_ne_bytes(), 20u32.to_ne_bytes()].concat(),
    );
}
