//! Component-local World GPU rows: a rare component must not inherit the
//! World's global entity high-water mark from a much larger population.

use pulsar_scenedb::gpu::{
    EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::{component::component_id, World};
use pulsar_scenedb_derive::SceneStore;
use std::sync::Arc;

const OBJECT_BUFFER_KEY: &str = "scenedb.test.component_rows.objects";
const LIGHT_BUFFER_KEY: &str = "scenedb.test.component_rows.lights";

#[repr(transparent)]
#[derive(Clone, Copy)]
struct ObjectGpuRow([u32; 4]);
// SAFETY: transparent over a fully initialized fixed-size integer array;
// there is no padding and every bit pattern is valid.
unsafe impl pulsar_scenedb::Pod for ObjectGpuRow {}
unsafe impl pulsar_scenedb::bytemuck::Zeroable for ObjectGpuRow {}
unsafe impl pulsar_scenedb::bytemuck::Pod for ObjectGpuRow {}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct LightGpuRow([u32; 32]);
// SAFETY: same representation argument as ObjectGpuRow.
unsafe impl pulsar_scenedb::Pod for LightGpuRow {}
unsafe impl pulsar_scenedb::bytemuck::Zeroable for LightGpuRow {}
unsafe impl pulsar_scenedb::bytemuck::Pod for LightGpuRow {}

#[derive(SceneStore, Clone, Copy)]
struct ObjectRow {
    #[gpu(buffer = "scenedb.test.component_rows.objects")]
    value: ObjectGpuRow,
}

#[derive(SceneStore, Clone, Copy)]
struct LightRow {
    // Deliberately 128 bytes: this is the expensive sparse-light failure
    // shape which exposed global Entity.index addressing in integration.
    #[gpu(buffer = "scenedb.test.component_rows.lights")]
    value: LightGpuRow,
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
        label: Some("scenedb-component-local-world-rows-test"),
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
        label: Some("component-local-row-readback"),
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

fn first_u32(bytes: &[u8]) -> u32 {
    u32::from_ne_bytes(bytes[..4].try_into().unwrap())
}

#[test]
fn rare_component_uses_and_reuses_its_own_row_zero_without_buffer_growth() {
    let ctx = test_context();
    let mut store = SceneGpuStore::new(&ctx, scene_cfg());
    ObjectRow::register_gpu_columns_growable(&mut store, 2, ctx.device());
    LightRow::register_gpu_columns_growable(&mut store, 1, ctx.device());
    let store = Arc::new(store);
    let mirror = GpuMirrorHandle::new(Arc::clone(&store), Arc::clone(ctx.queue()));
    let mut world = World::new();
    world.attach_gpu_mirror(mirror.clone());

    let mut entities = Vec::new();
    for value in 0..128_u32 {
        let entity = world.spawn();
        world.insert(
            entity,
            ObjectRow {
                value: ObjectGpuRow([value, 0, 0, 0]),
            },
        );
        entities.push(entity);
    }
    let first_light_entity = *entities.last().unwrap();
    assert_eq!(first_light_entity.index(), 127);
    world.insert(
        first_light_entity,
        LightRow {
            value: LightGpuRow([0xA11C_E001; 32]),
        },
    );

    assert_eq!(mirror.gpu_row::<ObjectRow>(first_light_entity), Some(127));
    assert_eq!(mirror.gpu_row::<LightRow>(first_light_entity), Some(0));
    assert_eq!(world.gpu_row::<LightRow>(first_light_entity), Some(0));
    assert_eq!(mirror.gpu_live_count::<LightRow>(), 1);
    assert_eq!(mirror.gpu_row_span::<LightRow>(), 1);

    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");
    let (light_buffer, epoch, desc) = store
        .gpu_buffer_snapshot_for_key(LIGHT_BUFFER_KEY)
        .expect("named light buffer");
    assert_eq!(desc.buffer_key, Some(LIGHT_BUFFER_KEY));
    assert_eq!(epoch, 0, "row 0 fits the one-row initial allocation");
    assert_eq!(light_buffer.size(), 128);
    assert_eq!(first_u32(&readback(&ctx, &light_buffer, 0, 128)), 0xA11C_E001);

    assert!(world.remove::<LightRow>(first_light_entity).is_some());
    assert_eq!(mirror.gpu_row::<LightRow>(first_light_entity), None);
    assert_eq!(mirror.gpu_live_count::<LightRow>(), 0);
    assert_eq!(mirror.gpu_row_span::<LightRow>(), 0);
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let second_light_entity = world.spawn();
    assert!(second_light_entity.index() >= 128);
    world.insert(
        second_light_entity,
        LightRow {
            value: LightGpuRow([0xB22D_E002; 32]),
        },
    );
    assert_eq!(mirror.gpu_row::<LightRow>(second_light_entity), Some(0));
    world.flush_gpu_mirror(ctx.queue()).expect("mirror attached");

    let (light_buffer, epoch_after_reuse, _) = store
        .gpu_buffer_snapshot_for_key(LIGHT_BUFFER_KEY)
        .expect("named light buffer after reuse");
    assert_eq!(epoch_after_reuse, epoch, "row reuse keeps buffer identity epoch");
    assert_eq!(light_buffer.size(), 128);
    assert_eq!(first_u32(&readback(&ctx, &light_buffer, 0, 128)), 0xB22D_E002);

    // The much larger object allocation grew independently; its named
    // reflection identity remains available and does not alias the light.
    let (object_buffer, object_epoch, desc) = store
        .gpu_buffer_snapshot_for_key(OBJECT_BUFFER_KEY)
        .expect("named object buffer");
    assert_eq!(desc.buffer_key, Some(OBJECT_BUFFER_KEY));
    assert!(object_epoch > 0);
    assert!(object_buffer.size() >= 128 * 16);

    mirror
        .reserve_gpu_component_capacity::<LightRow>(8)
        .expect("component-local reserve");
    let (reserved_light, reserved_epoch, _) = store
        .gpu_buffer_snapshot_for_key(LIGHT_BUFFER_KEY)
        .expect("reserved light buffer");
    assert_eq!(reserved_light.size(), 8 * 128);
    assert!(reserved_epoch > epoch_after_reuse);
    let (_, object_epoch_after_light_reserve, _) = store
        .gpu_buffer_snapshot_for_key(OBJECT_BUFFER_KEY)
        .expect("object buffer after light-only reserve");
    assert_eq!(
        object_epoch_after_light_reserve, object_epoch,
        "reserving lights must not reallocate the object buffer",
    );

    assert!(mirror.shrink_gpu_component_to_fit::<LightRow>(1.0));
    let (shrunk_light, shrunk_epoch, _) = store
        .gpu_buffer_snapshot_for_key(LIGHT_BUFFER_KEY)
        .expect("shrunk light buffer");
    assert_eq!(shrunk_light.size(), 128);
    assert!(shrunk_epoch > reserved_epoch);
    let (presence, _) = store
        .component_presence_buffer_snapshot_for_id(component_id::<LightRow>())
        .expect("light presence buffer");
    assert_eq!(presence.size(), 4);
}
