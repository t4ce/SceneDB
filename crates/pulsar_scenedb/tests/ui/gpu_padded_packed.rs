use pulsar_scenedb_derive::SceneStore;

// The generated repr(C) shader row would contain three implicit bytes
// between `tag` and `value`. SceneDB requires those bytes to be represented
// explicitly instead of reading/comparing uninitialized padding.
#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed)]
struct ImplicitlyPaddedGpuRow {
    #[gpu]
    tag: u8,
    #[gpu]
    value: u32,
}

fn main() {}
