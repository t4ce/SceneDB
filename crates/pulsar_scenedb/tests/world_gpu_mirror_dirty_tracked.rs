//! Proves the two behaviors #29 adds to `gpu::world_mirror`:
//!
//! 1. `#[gpu(mirror = Once)]` fields write on the first insert and never
//!    again — a later `world.insert()` re-inserting the same component
//!    (updating some other field) must NOT re-touch a `Once`-mode field's
//!    GPU buffer.
//! 2. `#[gpu]` (`DirtyTracked`, the default) fields defer their write --
//!    `World::insert` alone does not reach the GPU; `World::flush_gpu_mirror`
//!    does, coalesced.
//!
//! As of SceneDB#39, (1) also defers: `Once`-mode fields are queued on
//! insert and uploaded by the next `flush_gpu_mirror` call, the same as
//! `DirtyTracked` fields, instead of writing immediately inline with
//! `insert`. "Never touched again after the first insert" still holds --
//! only *when* that one write actually reaches the GPU changed.

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
    .expect("no adapter — GPU tests need a local GPU");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("scenedb-world-gpu-mirror-dirty-tracked-test"),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(Arc::new(device), Arc::new(queue))
}

fn readback_u32(ctx: &EngineGpuContext, buf: &wgpu::Buffer, row: u64) -> u32 {
    let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device().create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, row * 4, &staging, 0, 4);
    ctx.queue().submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    ctx.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    u32::from_ne_bytes(data.try_into().unwrap())
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

#[derive(SceneStore, Clone, Copy)]
struct MixedModeComponent {
    #[gpu(mirror = Once)]
    mesh_id: u32, // set once at spawn, never changes
    #[gpu]
    hp: u32, // DirtyTracked (default) -- changes every frame in a real game
}

#[derive(SceneStore, Clone, Copy)]
#[repr(C)]
struct DifferentialComponent {
    #[gpu]
    transform_version: u32,
    cpu_debug_tag: u32,
    #[gpu]
    material_version: u32,
    #[gpu(mirror = Once)]
    authored_mesh: u32,
}

#[derive(SceneStore, Clone, Copy)]
struct NanComponent {
    #[gpu]
    value: f32,
}

#[test]
fn raw_mutable_access_is_rejected_and_copy_edit_dispatches_the_mirror() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    DifferentialComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));
    let entity = world.spawn();
    world.insert(
        entity,
        DifferentialComponent {
            transform_version: 10,
            cpu_debug_tag: 1,
            material_version: 20,
            authored_mesh: 30,
        },
    );
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    let get_mut = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = world.get_mut::<DifferentialComponent>(entity);
    }));
    assert!(get_mut.is_err(), "get_mut bypassed GPU mirror dispatch");

    let query_mut = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = world.query_mut::<&mut DifferentialComponent>();
    }));
    assert!(query_mut.is_err(), "query_mut bypassed GPU mirror dispatch");

    world
        .edit::<DifferentialComponent, _>(entity, |component| {
            component.transform_version += 1;
        })
        .unwrap();
    let edited = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((edited.ranges, edited.bytes), (1, 4));
}

#[test]
fn in_place_replacement_dirties_only_changed_gpu_fields() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    DifferentialComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let columns = DifferentialComponent::gpu_columns();
    let authored_mesh_id = columns
        .iter()
        .find(|column| column.buffer_name == "authored_mesh")
        .unwrap()
        .field_token
        .id();
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));
    let entity = world.spawn();

    world.insert(
        entity,
        DifferentialComponent {
            transform_version: 10,
            cpu_debug_tag: 1,
            material_version: 20,
            authored_mesh: 30,
        },
    );
    world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(store.once_pending_count_for_id(authored_mesh_id), Some(0));

    // An exact replacement has an old component to compare against but no
    // changed GPU payload, so it must queue no value, presence, or generation
    // row at all.
    world.insert(
        entity,
        DifferentialComponent {
            transform_version: 10,
            cpu_debug_tag: 1,
            material_version: 20,
            authored_mesh: 30,
        },
    );
    let unchanged = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((unchanged.ranges, unchanged.bytes), (0, 0));

    // CPU-only state remains authoritative in World but does not touch any
    // GPU partner.
    world.insert(
        entity,
        DifferentialComponent {
            transform_version: 10,
            cpu_debug_tag: 2,
            material_version: 20,
            authored_mesh: 30,
        },
    );
    let cpu_only = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((cpu_only.ranges, cpu_only.bytes), (0, 0));

    // Once remains a presence-lifetime handoff, not a value-differential
    // field. Changing it in place neither queues a second handoff nor dirties
    // another column.
    world.insert(
        entity,
        DifferentialComponent {
            transform_version: 10,
            cpu_debug_tag: 2,
            material_version: 20,
            authored_mesh: 999,
        },
    );
    assert_eq!(store.once_pending_count_for_id(authored_mesh_id), Some(0));
    let once_update = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((once_update.ranges, once_update.bytes), (0, 0));

    // One changed u32 field produces exactly one four-byte dirty row. The
    // other DirtyTracked column is compared directly and skipped.
    world.insert(
        entity,
        DifferentialComponent {
            transform_version: 11,
            cpu_debug_tag: 2,
            material_version: 20,
            authored_mesh: 999,
        },
    );
    let one_gpu_field = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((one_gpu_field.ranges, one_gpu_field.bytes), (1, 4));
}

#[test]
fn differential_comparison_uses_nan_bits_not_float_equality() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    NanComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));
    let entity = world.spawn();

    let first_nan = f32::from_bits(0x7fc0_0001);
    world.insert(entity, NanComponent { value: first_nan });
    world.flush_gpu_mirror(ctx.queue()).unwrap();

    // NaN != NaN under PartialEq, but an identical shader-row bit pattern is
    // unchanged and must not upload.
    world.insert(
        entity,
        NanComponent {
            value: f32::from_bits(0x7fc0_0001),
        },
    );
    let same_payload = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((same_payload.ranges, same_payload.bytes), (0, 0));

    // A distinct NaN payload is a distinct GPU value even though both are
    // NaN numerically, so it dirties exactly one f32 row.
    world.insert(
        entity,
        NanComponent {
            value: f32::from_bits(0x7fc0_0002),
        },
    );
    let different_payload = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!((different_payload.ranges, different_payload.bytes), (1, 4));
}

#[test]
fn once_mode_field_never_rewrites_after_the_first_insert() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    MixedModeComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));

    let entity = world.spawn();

    let columns = MixedModeComponent::gpu_columns();
    let mesh_id_col = columns.iter().find(|c| c.buffer_name == "mesh_id").unwrap();
    let mesh_id_field_id = mesh_id_col.field_token.id();

    world.insert(
        entity,
        MixedModeComponent {
            mesh_id: 42,
            hp: 100,
        },
    );
    let row = world
        .gpu_row::<MixedModeComponent>(entity)
        .expect("component GPU row") as u64;
    assert_eq!(
        store.once_pending_count_for_id(mesh_id_field_id),
        Some(1),
        "Once keeps only a transient pending handoff, not a capacity-sized CPU shadow",
    );
    // SceneDB#39: Once-mode writes are now queued, not immediate -- a flush
    // is required before this reaches the GPU, same as DirtyTracked fields.
    world
        .flush_gpu_mirror(ctx.queue())
        .expect("mirror attached");
    let mut got = 0u32;
    store.with_once_buffer_for_id(mesh_id_field_id, &mut |buf| {
        got = readback_u32(&ctx, buf, row)
    });
    assert_eq!(
        got, 42,
        "Once field must have written by the first flush after its first insert"
    );
    assert_eq!(store.once_pending_count_for_id(mesh_id_field_id), Some(0));

    // Re-insert (an in-place update -- entity already has this component):
    // mesh_id changes in the CPU-side value, but the GPU buffer must NOT
    // reflect it -- Once means "queued once at the first insert and never
    // touched again," not "immune to flush timing."
    world.insert(
        entity,
        MixedModeComponent {
            mesh_id: 999,
            hp: 50,
        },
    );
    world
        .flush_gpu_mirror(ctx.queue())
        .expect("mirror attached");
    let mut got_after = 0u32;
    store.with_once_buffer_for_id(mesh_id_field_id, &mut |buf| {
        got_after = readback_u32(&ctx, buf, row)
    });
    assert_eq!(
        got_after, 42,
        "Once field must NOT re-write on a later update, even though the CPU value changed"
    );
}

#[test]
fn dirty_tracked_field_defers_until_flush_gpu_mirror() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    MixedModeComponent::register_gpu_columns_growable(&mut store, 8, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));

    let entity = world.spawn();
    let hp_field_id = MixedModeComponent::gpu_columns()
        .iter()
        .find(|c| c.buffer_name == "hp")
        .unwrap()
        .field_token
        .id();

    world.insert(
        entity,
        MixedModeComponent {
            mesh_id: 1,
            hp: 100,
        },
    );
    let row = world
        .gpu_row::<MixedModeComponent>(entity)
        .expect("component GPU row") as u64;

    let mut before_flush = 0u32;
    store.with_dirty_tracked_buffer_for_id(hp_field_id, &mut |buf| {
        before_flush = readback_u32(&ctx, buf, row)
    });
    assert_eq!(
        before_flush, 0,
        "DirtyTracked field must NOT be on the GPU yet -- insert alone must not reach it"
    );

    // Several updates before any flush -- only the LAST value should matter
    // once flushed, and only ONE upload should be needed regardless of how
    // many times it changed in between.
    world.insert(entity, MixedModeComponent { mesh_id: 1, hp: 80 });
    world.insert(entity, MixedModeComponent { mesh_id: 1, hp: 60 });
    world.insert(entity, MixedModeComponent { mesh_id: 1, hp: 42 });

    let stats = world
        .flush_gpu_mirror(ctx.queue())
        .expect("mirror attached");
    assert!(stats.ranges >= 1, "flush must have uploaded something");

    let mut after_flush = 0u32;
    store.with_dirty_tracked_buffer_for_id(hp_field_id, &mut |buf| {
        after_flush = readback_u32(&ctx, buf, row)
    });
    assert_eq!(
        after_flush, 42,
        "flush must upload the LATEST value, not an intermediate one"
    );

    // A second flush with nothing newly dirty must be a true no-op.
    let stats2 = world.flush_gpu_mirror(ctx.queue()).unwrap();
    assert_eq!(stats2.ranges, 0);
    assert_eq!(stats2.bytes, 0);
}

#[test]
fn dirty_tracked_writes_across_multiple_entities_coalesce_on_flush() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    MixedModeComponent::register_gpu_columns_growable(&mut store, 16, ctx.device());
    let store = Arc::new(store);

    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(
        Arc::clone(&store),
        Arc::clone(ctx.queue()),
    ));

    let hp_field_id = MixedModeComponent::gpu_columns()
        .iter()
        .find(|c| c.buffer_name == "hp")
        .unwrap()
        .field_token
        .id();

    // 5 adjacent entities (rows 0..5) -- must coalesce into ONE range.
    let mut entities = Vec::new();
    for i in 0..5u32 {
        let e = world.spawn();
        world.insert(
            e,
            MixedModeComponent {
                mesh_id: 0,
                hp: i * 10,
            },
        );
        entities.push(e);
    }
    let stats = world.flush_gpu_mirror(ctx.queue()).unwrap();
    // One coalesced range each for hp, mesh_id, component presence, and
    // entity generation. Lifecycle validity uploads are intentionally part
    // of the returned total rather than hidden bookkeeping.
    assert_eq!(
        stats.ranges, 4,
        "5 adjacent rows must coalesce once per value and validity column"
    );

    for (i, e) in entities.iter().enumerate() {
        let mut got = 0u32;
        store.with_dirty_tracked_buffer_for_id(hp_field_id, &mut |buf| {
            got = readback_u32(&ctx, buf, e.index() as u64)
        });
        assert_eq!(got, (i as u32) * 10);
    }
}
