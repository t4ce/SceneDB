use std::collections::HashMap;

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{parse::Parse, Data, DeriveInput, Fields, Ident, LitStr, Type};

use crate::cell::generate_scene_column_set;
use crate::gpu::generate_gpu_column_set;

// ── #[gpu] attribute parsing ──────────────────────────────────────────────

pub struct GpuAttr {
    pub mirror_mode: Option<MirrorModeAttr>,
    /// Explicit renderer-facing identity for this field's destination GPU
    /// buffer. Kept as a `LitStr` so code generation emits a true
    /// `&'static str`, rather than constructing an owned name at runtime.
    pub buffer_key: Option<LitStr>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MirrorModeAttr {
    DirtyTracked,
    Once,
}

impl Parse for GpuAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut mirror_mode = None;
        let mut buffer_key = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let _: syn::Token![=] = input.parse()?;

            if key == "mirror" {
                if mirror_mode.is_some() {
                    return Err(syn::Error::new(key.span(), "duplicate `mirror` option"));
                }
                let mode: Ident = input.parse()?;
                mirror_mode = Some(match mode.to_string().as_str() {
                    "DirtyTracked" => MirrorModeAttr::DirtyTracked,
                    "Once" => MirrorModeAttr::Once,
                    _ => {
                        return Err(syn::Error::new(
                            mode.span(),
                            "expected DirtyTracked or Once",
                        ))
                    }
                });
            } else if key == "buffer" {
                if buffer_key.is_some() {
                    return Err(syn::Error::new(key.span(), "duplicate `buffer` option"));
                }
                let name: LitStr = input.parse().map_err(|_| {
                    syn::Error::new(
                        input.span(),
                        "expected a string literal (e.g. `buffer = \"general_mesh_buf\"`)",
                    )
                })?;
                if name.value().is_empty() {
                    return Err(syn::Error::new(
                        name.span(),
                        "GPU buffer name cannot be empty",
                    ));
                }
                buffer_key = Some(name);
            } else {
                return Err(syn::Error::new(key.span(), "expected `mirror` or `buffer`"));
            }

            if input.is_empty() {
                break;
            }
            let _: syn::Token![,] = input.parse()?;
        }

        Ok(GpuAttr {
            mirror_mode,
            buffer_key,
        })
    }
}

// ── Struct-level `#[gpu(layout = packed)]` attribute parsing ───────────────
//
// A separate, struct-level use of the same `gpu` attribute name as the
// per-field one above -- no ambiguity, since `syn`/the derive macro reads
// struct attrs (`DeriveInput::attrs`) and field attrs (`Field::attrs`)
// through entirely separate code paths.

pub struct StructGpuAttr {
    pub layout_packed: bool,
}

impl Parse for StructGpuAttr {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(StructGpuAttr {
                layout_packed: false,
            });
        }
        let key: Ident = input.parse()?;
        if key != "layout" {
            return Err(syn::Error::new(
                key.span(),
                "expected `layout` (e.g. `#[gpu(layout = packed)]`)",
            ));
        }
        let _: syn::Token![=] = input.parse()?;
        let value: Ident = input.parse()?;
        if value != "packed" {
            return Err(syn::Error::new(
                value.span(),
                "expected `packed` -- the only supported layout today",
            ));
        }
        Ok(StructGpuAttr {
            layout_packed: true,
        })
    }
}

/// Scans a struct's own attributes (not its fields') for `#[gpu(layout = packed)]`.
///
/// Packed layout groups every `#[gpu]` field into ONE GPU buffer (a single
/// interleaved record per row) instead of the default one-buffer-per-field
/// split -- for structs, like a renderer's per-instance GPU record, whose
/// `#[gpu]` fields are always read together and were never independent
/// columns to begin with. Deliberately scoped to the World-mirror path only
/// (`register_gpu_columns_growable` + `World::insert`'s automatic dispatch):
/// it does NOT change `gpu_columns()`, `write_gpu`, or the fixed
/// `register_gpu_columns` at all -- those stay exactly as they are for
/// EVERY `#[derive(SceneStore)]` type, packed or not, because the
/// cell-mirrored path's dirty-tracked boundary sync reads FROM CellStorage's
/// own per-field SoA columns, which packing has no relationship to (packing
/// only changes what shape of buffer the data is written *into* on the GPU
/// side, not how it's stored on the CPU side). Requires at least one
/// `#[gpu]` field to have any effect -- a packed struct with none behaves
/// identically to one without the attribute at all (nothing to pack).
pub fn struct_is_packed(attrs: &[syn::Attribute]) -> bool {
    // Lenient on parse failure (a bare `#[gpu]` with no `(...)` at all, or
    // unrecognized content). This is the historical struct-level behavior;
    // per-field options are parsed strictly because silently ignoring a
    // misspelled stable buffer identity would register the wrong physical
    // column.
    attrs.iter().any(|attr| {
        attr.path().is_ident("gpu")
            && attr
                .parse_args::<StructGpuAttr>()
                .map(|parsed| parsed.layout_packed)
                .unwrap_or(false)
    })
}

// ── Per-field metadata ────────────────────────────────────────────────────

pub struct FieldInfo {
    pub ident: Ident,
    pub ty: Type,
    pub is_gpu: bool,
    pub mirror_mode: MirrorModeAttr,
    /// Optional explicit stable destination key from
    /// `#[gpu(buffer = "...")]`. `None` preserves the historical behavior:
    /// the generated wrapper remains this field's private physical identity.
    /// `GpuColumnDesc::buffer_name` remains the Rust field/display name in
    /// both cases; the new `buffer_key` metadata is populated only for this
    /// explicit opt-in, preventing common bare field names from aliasing.
    pub gpu_buffer_key: Option<LitStr>,
    /// Present iff `is_gpu`. `ComponentId`/`TypeToken` (this crate's GPU
    /// buffer + CPU-column keys) are derived from a Rust `TypeId`, globally
    /// — keyed by the field's own raw type, they carry no notion of which
    /// *struct* the field belongs to. Two different `#[derive(SceneStore)]`
    /// types both having, say, an `f32` field marked `#[gpu]` would
    /// otherwise resolve to the exact same `ComponentId`, and the second
    /// type's `register_gpu_buffer::<f32>()` call would silently replace
    /// the first's GPU buffer outright (`SceneGpuStore::register_gpu_buffer`
    /// does a plain `HashMap::insert`, no collision check) — not a data
    /// corruption in the row-range sense (each cell's rows are disjoint,
    /// per `RegionPool`), but a semantic one: "the roughness buffer" and
    /// "the intensity buffer" would silently be the same physical buffer,
    /// interleaved by row region, which is never what marking two
    /// unrelated fields `#[gpu]` is asking for.
    ///
    /// Fixed by generating one `#[repr(transparent)]` newtype wrapper per
    /// `#[gpu]` field (`__ScenedbGpuCol_<Struct>_<Field>`, byte-identical
    /// to the field's own type) and using *that* — not the raw field type
    /// — as the column's registered type everywhere: `GpuColumnDesc::
    /// field_token`, the `write_gpu`-generated `component_id::<_>()` call,
    /// and (when the `gpu` feature is on) the `SceneColumnSet`-generated
    /// `CellType` column token. A wrapper's own `TypeId` is unique to its
    /// (struct, field) pair by construction, so two `#[gpu] f32` fields on
    /// different structs get two distinct CPU-column `ComponentId`s even
    /// though their underlying data is the same shape. An explicit
    /// `buffer = "..."` may subsequently alias compatible wrappers to one
    /// canonical GPU allocation through `SceneGpuStore`'s descriptor
    /// registry; the wrapper identity itself remains stable for CellStorage.
    pub gpu_wrapper: Option<Ident>,
}

impl FieldInfo {
    /// Stable cross-component buffer identity, when the field explicitly
    /// opted into one with `#[gpu(buffer = "...")]`.
    ///
    /// A bare `#[gpu]` deliberately returns `None`: ordinary field names
    /// such as `value` or `position` are display names, not global buffer
    /// identities. Treating those historical names as keys would make
    /// unrelated components alias merely because their Rust field names
    /// happen to match.
    pub fn gpu_buffer_key(&self) -> Option<&LitStr> {
        self.gpu_buffer_key.as_ref()
    }
}

fn validate_gpu_buffer_names(gpu_fields: &[&FieldInfo], is_packed: bool) -> syn::Result<()> {
    let mut explicit_names: HashMap<String, Span> = HashMap::new();

    for field in gpu_fields {
        let Some(name) = field.gpu_buffer_key() else {
            continue;
        };

        if is_packed {
            return Err(syn::Error::new(
                name.span(),
                "`#[gpu(buffer = \"...\")]` is not supported with `#[gpu(layout = packed)]`: \
                 the packed World mirror is one interleaved physical buffer, so a per-field \
                 name would not identify the buffer that is actually registered",
            ));
        }

        if let Some(first_span) = explicit_names.insert(name.value(), name.span()) {
            let mut error = syn::Error::new(
                name.span(),
                format!(
                    "duplicate GPU buffer name `{}` within one SceneStore type; named fields \
                     require distinct buffers (packed/grouped named buffers are not supported)",
                    name.value()
                ),
            );
            error.combine(syn::Error::new(
                first_span,
                "first use of this GPU buffer name",
            ));
            return Err(error);
        }
    }

    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(ds) => match &ds.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "SceneStore requires named fields",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                name,
                "SceneStore only supports structs",
            ))
        }
    };

    let mut field_infos: Vec<FieldInfo> = Vec::new();
    for field in fields {
        let ident = field.ident.as_ref().unwrap().clone();
        let ty = field.ty.clone();
        let mut is_gpu = false;
        let mut mirror_mode = MirrorModeAttr::DirtyTracked;
        let mut mirror_mode_explicit = false;
        let mut gpu_buffer_key = None;

        for attr in &field.attrs {
            if attr.path().is_ident("gpu") {
                is_gpu = true;
                // `Attribute::parse_args` intentionally rejects path-only
                // attributes, so preserve bare `#[gpu]` as the default
                // DirtyTracked/no-key form and parse strictly only when an
                // argument list is actually present.
                let gpu_attr = if matches!(attr.meta, syn::Meta::Path(_)) {
                    GpuAttr {
                        mirror_mode: None,
                        buffer_key: None,
                    }
                } else {
                    attr.parse_args::<GpuAttr>()?
                };
                if let Some(mode) = gpu_attr.mirror_mode {
                    if mirror_mode_explicit {
                        return Err(syn::Error::new_spanned(
                            attr,
                            "duplicate `mirror` option across `#[gpu]` attributes",
                        ));
                    }
                    mirror_mode = mode;
                    mirror_mode_explicit = true;
                }
                if let Some(buffer_key) = gpu_attr.buffer_key {
                    if gpu_buffer_key.is_some() {
                        return Err(syn::Error::new_spanned(
                            attr,
                            "duplicate `buffer` option across `#[gpu]` attributes",
                        ));
                    }
                    gpu_buffer_key = Some(buffer_key);
                }
            }
        }

        // The wrapper ident itself is intentionally unique to (struct,
        // field). GPU-bearing generic SceneStore types are rejected below:
        // an ident cannot encode a Rust monomorph, and inventory also needs
        // a concrete component identity. CPU-only generic types never emit
        // this wrapper and remain supported.
        let gpu_wrapper = is_gpu
            .then(|| Ident::new(&format!("__ScenedbGpuCol_{}_{}", name, ident), ident.span()));

        field_infos.push(FieldInfo {
            ident,
            ty,
            is_gpu,
            mirror_mode,
            gpu_buffer_key,
            gpu_wrapper,
        });
    }

    if field_infos.is_empty() {
        return Err(syn::Error::new_spanned(
            name,
            "SceneStore requires at least one field",
        ));
    }

    let gpu_fields: Vec<&FieldInfo> = field_infos.iter().filter(|f| f.is_gpu).collect();

    if !gpu_fields.is_empty() && !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "`#[derive(SceneStore)]` does not yet support `#[gpu]` fields on generic structs: \
             GPU inventory/partner reflection requires a concrete component type, and explicit \
             per-monomorph registration is not yet provided; use a concrete SceneStore wrapper \
             or remove `#[gpu]` (generic CPU-only SceneStore types remain supported)",
        ));
    }

    let is_packed = struct_is_packed(&input.attrs);
    validate_gpu_buffer_names(&gpu_fields, is_packed)?;

    // Two `SceneColumnSet` impls, `cfg`-split on the `gpu` feature: with it
    // on, `#[gpu]` fields' CellType column tokens must match the wrapper
    // types `write_gpu`/`GpuColumnDesc` use (see `gpu_wrapper`'s doc) or
    // `cell.column_for_mut::<Wrapper>()` would find no column; with it off
    // there is no GPU column concept at all, so every field (including ones
    // marked `#[gpu]`, which is a no-op without the feature) keeps its own
    // natural type -- unchanged from before this fix.
    let scene_column_set_gpu = generate_scene_column_set(
        name,
        &impl_generics,
        &ty_generics,
        where_clause,
        &field_infos,
        true,
    );
    let scene_column_set_no_gpu = generate_scene_column_set(
        name,
        &impl_generics,
        &ty_generics,
        where_clause,
        &field_infos,
        false,
    );

    let gpu_wrapper_defs: Vec<TokenStream> = gpu_fields
        .iter()
        .map(|f| {
            let wrapper = f
                .gpu_wrapper
                .as_ref()
                .expect("gpu field has a wrapper ident");
            let ty = &f.ty;
            quote! {
                // Byte-identical to #ty (repr(transparent), single field) --
                // exists solely to give this field's GPU column a TypeId
                // unique to (this struct, this field). See `FieldInfo::
                // gpu_wrapper`'s doc for why that's load-bearing.
                #[doc(hidden)]
                #[allow(non_camel_case_types)]
                #[repr(transparent)]
                #[derive(
                    Clone,
                    Copy,
                    ::pulsar_scenedb::bytemuck::Zeroable,
                    ::pulsar_scenedb::bytemuck::Pod,
                )]
                pub struct #wrapper(pub #ty);
                unsafe impl ::pulsar_scenedb::page::Pod for #wrapper {}
            }
        })
        .collect();

    let gpu_column_set = generate_gpu_column_set(
        name,
        &impl_generics,
        &ty_generics,
        where_clause,
        &gpu_fields,
        is_packed,
    );
    // NOTE: HasTypeToken is NOT generated here — the blanket impl in
    // `pulsar_scenedb::token` covers `T: Pod + 'static`, which our Pod impl
    // satisfies.  An explicit impl would conflict.

    Ok(quote! {
        #[cfg(feature = "gpu")]
        const _: () = {
            #(#gpu_wrapper_defs)*
            #scene_column_set_gpu
            #gpu_column_set
        };
        #[cfg(not(feature = "gpu"))]
        #scene_column_set_no_gpu
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn gpu_attr_parses_named_buffer_and_composed_options() {
        let named: GpuAttr = syn::parse_str(r#"buffer = "general_mesh_buf""#).expect("buffer");
        assert_eq!(
            named.buffer_key.as_ref().map(LitStr::value),
            Some("general_mesh_buf".to_owned())
        );
        assert!(named.mirror_mode.is_none());

        let composed: GpuAttr = syn::parse_str(r#"mirror = Once, buffer = "general_mesh_buf","#)
            .expect("composed options with trailing comma");
        assert!(matches!(composed.mirror_mode, Some(MirrorModeAttr::Once)));
        assert_eq!(
            composed.buffer_key.as_ref().map(LitStr::value),
            Some("general_mesh_buf".to_owned())
        );

        let reverse_order: GpuAttr =
            syn::parse_str(r#"buffer = "general_mesh_buf", mirror = DirtyTracked"#)
                .expect("options in reverse order");
        assert!(matches!(
            reverse_order.mirror_mode,
            Some(MirrorModeAttr::DirtyTracked)
        ));
    }

    #[test]
    fn gpu_attr_rejects_invalid_or_ambiguous_buffer_options() {
        for (input, expected) in [
            (r#"buffer = """#, "GPU buffer name cannot be empty"),
            ("buffer = general_mesh_buf", "expected a string literal"),
            (r#"buffer = "a", buffer = "b""#, "duplicate `buffer` option"),
            (r#"destination = "a""#, "expected `mirror` or `buffer`"),
        ] {
            let error = syn::parse_str::<GpuAttr>(input)
                .err()
                .unwrap_or_else(|| panic!("`{input}` unexpectedly parsed"));
            assert!(
                error.to_string().contains(expected),
                "`{input}`: expected `{expected}`, got `{error}`"
            );
        }
    }

    #[test]
    fn derive_rejects_duplicate_explicit_buffer_keys_within_one_type() {
        let input: DeriveInput = parse_quote! {
            struct DuplicateNames {
                #[gpu(buffer = "shared")]
                first: u32,
                #[gpu(buffer = "shared")]
                second: u32,
            }
        };
        let error = expand(input).expect_err("duplicate explicit keys must fail");
        assert!(error
            .to_string()
            .contains("duplicate GPU buffer name `shared`"));
    }

    #[test]
    fn bare_field_name_is_not_an_implicit_global_buffer_key() {
        let input: DeriveInput = parse_quote! {
            struct ExplicitAndDisplayName {
                #[gpu(buffer = "second")]
                first: u32,
                #[gpu]
                second: u32,
            }
        };
        expand(input).expect("a bare field name is display metadata, not a shared key");
    }

    #[test]
    fn derive_rejects_named_fields_in_packed_world_layout() {
        let input: DeriveInput = parse_quote! {
            #[gpu(layout = packed)]
            struct NamedPacked {
                #[gpu(buffer = "not_the_actual_packed_buffer")]
                value: u32,
            }
        };
        let error = expand(input).expect_err("named packed field must fail");
        assert!(error
            .to_string()
            .contains("not supported with `#[gpu(layout = packed)]`"));
    }

    #[test]
    fn derive_rejects_gpu_fields_on_generic_scene_store_types() {
        let input: DeriveInput = parse_quote! {
            struct GenericGpu<T> {
                #[gpu(buffer = "generic_values")]
                value: T,
            }
        };
        let error = expand(input).expect_err("generic GPU identity is not monomorphic");
        assert!(error
            .to_string()
            .contains("GPU inventory/partner reflection requires a concrete component type"));
    }

    #[test]
    fn generic_cpu_only_scene_store_types_remain_supported() {
        let input: DeriveInput = parse_quote! {
            struct GenericCpu<T: ::pulsar_scenedb::page::Pod + 'static> {
                value: T,
            }
        };
        expand(input).expect("generic CPU-only SceneStore must remain supported");
    }
}
