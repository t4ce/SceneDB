#[test]
fn packed_shader_rows_reject_implicit_padding() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/gpu_padded_packed.rs");
}
