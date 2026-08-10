//! Cargo-level regressions for `SceneStore`'s downstream feature boundary.
//!
//! Proc-macro output is compiled in the consuming crate, so an ordinary
//! in-workspace derive test cannot prove that the expansion is independent of
//! that crate's feature names. These two temporary crates intentionally have
//! no `[features]` table and no direct `wgpu`, `bytemuck`, or reflection
//! dependencies.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct TempWorkspace(PathBuf);

impl TempWorkspace {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must follow the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "pulsar-scenedb-derive-downstream-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create downstream derive test workspace");
        Self(path)
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn manifest_path_literal(path: &Path) -> String {
    format!("{:?}", path.to_string_lossy())
}

fn write_case(root: &Path, name: &str, manifest: &str, source: &str) -> PathBuf {
    let case = root.join(name);
    fs::create_dir_all(case.join("src")).expect("create downstream fixture source directory");
    fs::write(case.join("Cargo.toml"), manifest).expect("write downstream fixture manifest");
    fs::write(case.join("src/lib.rs"), source).expect("write downstream fixture source");
    case.join("Cargo.toml")
}

fn cargo_check(manifest: &Path, target_dir: &Path) -> Output {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    Command::new(cargo)
        .args(["check", "--quiet", "--manifest-path"])
        .arg(manifest)
        .env("CARGO_TARGET_DIR", target_dir)
        .output()
        .expect("launch cargo check for downstream derive fixture")
}

fn assert_check_succeeds(label: &str, output: Output) {
    assert!(
        output.status.success(),
        "{label} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn derive_respects_dependency_gpu_capability_not_consumer_feature_names() {
    let scenedb_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let derive_dir = scenedb_dir
        .parent()
        .expect("pulsar_scenedb must live below the crates directory")
        .join("pulsar_scenedb_derive");
    let temp = TempWorkspace::new();
    let target_dir = scenedb_dir
        .parent()
        .and_then(Path::parent)
        .expect("SceneDB workspace root")
        .join("target/downstream-derive-contract");
    let scenedb_path = manifest_path_literal(&scenedb_dir);
    let derive_path = manifest_path_literal(&derive_dir);

    let gpu_manifest = format!(
        r#"[workspace]
resolver = "2"

[package]
name = "scenedb-downstream-gpu-contract"
version = "0.0.0"
edition = "2021"

[dependencies]
pulsar_scenedb = {{ path = {scenedb_path}, default-features = false, features = ["gpu"] }}
pulsar_scenedb_derive = {{ path = {derive_path} }}
"#,
    );
    let gpu_source = r#"
use pulsar_scenedb::{GpuColumnSet, MirrorMode, SceneColumnSet};
use pulsar_scenedb_derive::SceneStore;

#[derive(SceneStore, Clone, Copy)]
struct GpuPartner {
    #[gpu(mirror = Once, buffer = "fixture.general_mesh_buf")]
    mesh: u32,
    cpu_metadata: u64,
}

#[derive(SceneStore, Clone, Copy)]
#[gpu(layout = packed)]
struct PackedGpuPartner {
    #[gpu]
    first: u32,
    #[gpu]
    second: u32,
}

pub fn generated_contract_is_reachable() {
    let columns = <GpuPartner as GpuColumnSet>::gpu_columns();
    assert_eq!(columns[0].mode, MirrorMode::Once);
    let _ = <GpuPartner as SceneColumnSet>::cell_type();
    let _ = PackedGpuPartner::packed_gpu_component_id();
}
"#;
    let gpu_case = write_case(&temp.0, "gpu_dependency", &gpu_manifest, gpu_source);
    assert_check_succeeds(
        "GPU derive in a crate with no local `gpu` feature",
        cargo_check(&gpu_case, &target_dir),
    );

    let cpu_manifest = format!(
        r#"[workspace]
resolver = "2"

[package]
name = "scenedb-downstream-cpu-contract"
version = "0.0.0"
edition = "2021"

[dependencies]
pulsar_scenedb = {{ path = {scenedb_path}, default-features = false }}
pulsar_scenedb_derive = {{ path = {derive_path} }}
"#,
    );
    let cpu_source = r#"
use pulsar_scenedb::{Pod, SceneColumnSet};
use pulsar_scenedb_derive::SceneStore;

#[derive(SceneStore)]
struct CpuOnly<T: Pod + 'static> {
    value: T,
}

pub fn generic_cpu_contract_is_reachable() {
    let _ = <CpuOnly<u32> as SceneColumnSet>::cell_type();
}
"#;
    let cpu_case = write_case(&temp.0, "cpu_dependency", &cpu_manifest, cpu_source);
    assert_check_succeeds(
        "CPU-only derive against SceneDB without its `gpu` feature",
        cargo_check(&cpu_case, &target_dir),
    );
}
