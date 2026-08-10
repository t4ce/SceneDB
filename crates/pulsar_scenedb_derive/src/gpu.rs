use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::scene_store::{FieldInfo, MirrorModeAttr};

pub fn generate_gpu_column_set(
    name: &Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&syn::WhereClause>,
    gpu_fields: &[&FieldInfo],
    is_packed: bool,
) -> TokenStream {
    if gpu_fields.is_empty() {
        return quote! {
            unsafe impl #impl_generics ::pulsar_scenedb::GpuColumnSet for #name #ty_generics #where_clause {
                fn gpu_columns() -> Vec<::pulsar_scenedb::GpuColumnDesc> {
                    Vec::new()
                }
                fn write_gpu(
                    _store: &::pulsar_scenedb::gpu::SceneGpuStore,
                    _id: ::pulsar_scenedb::gpu::CellId,
                    _cell: &mut ::pulsar_scenedb::cell::CellStorage,
                    _handle: ::pulsar_scenedb::handle::Handle,
                    _data: &Self,
                    _phase: &impl ::pulsar_scenedb::gpu::SimulateWitness,
                ) {
                }
            }

            impl #impl_generics #name #ty_generics #where_clause {
                /// No `#[gpu]` fields on this type -- nothing to register.
                pub fn register_gpu_columns(
                    _store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                    _capacity: u32,
                    _device: &::wgpu::Device,
                ) {
                }

                /// No `#[gpu]` fields on this type -- nothing to register.
                pub fn register_gpu_columns_growable(
                    _store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                    _initial_capacity: u32,
                    _device: &::std::sync::Arc<::wgpu::Device>,
                ) {
                }
            }
        };
    }

    // Packed layout writes the whole record as one unit, so every `#[gpu]`
    // field in a packed struct must share one mirror mode -- there's no
    // sensible meaning for "some of this one packed write is Once and some
    // is DirtyTracked." Validated here, at macro-expansion time, as a real
    // diagnostic (not a macro-execution-time panic).
    if is_packed {
        let all_once = gpu_fields
            .iter()
            .all(|f| f.mirror_mode == MirrorModeAttr::Once);
        let all_dirty_tracked = gpu_fields
            .iter()
            .all(|f| f.mirror_mode == MirrorModeAttr::DirtyTracked);
        if !all_once && !all_dirty_tracked {
            return quote! {
                compile_error!(
                    "#[gpu(layout = packed)] requires every #[gpu] field to share the same \
                     mirror mode -- either all plain #[gpu] (DirtyTracked, the default) or all \
                     #[gpu(mirror = Once)]. Mixed modes have no sensible meaning for a packed \
                     column, since the whole record is written as one unit."
                );
            };
        }
    }
    let packed_is_once = is_packed
        && gpu_fields
            .iter()
            .all(|f| f.mirror_mode == MirrorModeAttr::Once);

    // Every `#[gpu]` field is stored (and registered as a GPU buffer) under
    // its own generated wrapper type, not its raw field type -- see
    // `FieldInfo::gpu_wrapper`'s doc for why: `TypeToken`/`ComponentId` are
    // TypeId-keyed globally, so two different structs' same-shaped `#[gpu]`
    // fields would otherwise silently alias one GPU buffer.
    let column_descs: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let field_ident = &f.ident;
            let field_ty = &f.ty;
            let wrapper = f
                .gpu_wrapper
                .as_ref()
                .expect("gpu field has a wrapper ident");
            let buffer_key = match f.gpu_buffer_key() {
                Some(name) => quote! { Some(#name) },
                None => quote! { None },
            };
            let mirror_mode = match f.mirror_mode {
                MirrorModeAttr::DirtyTracked => {
                    quote! { ::pulsar_scenedb::MirrorMode::DirtyTracked }
                }
                MirrorModeAttr::Once => {
                    quote! { ::pulsar_scenedb::MirrorMode::Once }
                }
            };
            quote! {
                ::pulsar_scenedb::GpuColumnDesc {
                    field_token: ::pulsar_scenedb::token::TypeToken::of::<#wrapper>(),
                    value_token: ::pulsar_scenedb::token::TypeToken::of::<#field_ty>(),
                    field_offset: ::std::mem::offset_of!(#name, #field_ident),
                    mode: #mirror_mode,
                    buffer_name: #field_name,
                    buffer_key: #buffer_key,
                }
            }
        })
        .collect();

    let write_arms: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let field_ident = &f.ident;
            let wrapper = f
                .gpu_wrapper
                .as_ref()
                .expect("gpu field has a wrapper ident");
            quote! {
                #field_name => {
                    let row = cell.row_of(handle).unwrap_or_else(|| {
                        panic!("write_gpu: handle {:?} not found in cell", handle);
                    }) as usize;
                    if let Some(col) = cell.column_for_mut::<#wrapper>() {
                        col[row] = #wrapper(data.#field_ident);
                    }
                    let comp_id = ::pulsar_scenedb::component::component_id::<#wrapper>();
                    store.mark_column_dirty(id, comp_id, row as u32);
                }
            }
        })
        .collect();

    let register_calls: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let buffer_label = f
                .gpu_buffer_key()
                .map(|key| key.value())
                .unwrap_or_else(|| format!("{}::{}", name, field_name));
            let wrapper = f
                .gpu_wrapper
                .as_ref()
                .expect("gpu field has a wrapper ident");
            quote! {
                store.register_gpu_buffer::<#wrapper>(capacity, device, #buffer_label);
            }
        })
        .collect();

    // DirtyTracked retains a row-indexed CPU shadow for arbitrary updates;
    // Once uses a transient handoff queue which is discarded after flush.
    // Keeping those registrations distinct is the memory contract promised
    // by MirrorMode::Once, not merely an implementation detail.
    let register_growable_calls: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_name = f.ident.to_string();
            let buffer_label = f
                .gpu_buffer_key()
                .map(|key| key.value())
                .unwrap_or_else(|| format!("{}::{}", name, field_name));
            let wrapper = f.gpu_wrapper.as_ref().expect("gpu field has a wrapper ident");
            match f.mirror_mode {
                MirrorModeAttr::DirtyTracked => quote! {
                    store.register_dirty_tracked_gpu_buffer::<#wrapper>(initial_capacity, device, #buffer_label);
                },
                MirrorModeAttr::Once => quote! {
                    store.register_once_gpu_buffer::<#wrapper>(initial_capacity, device, #buffer_label);
                },
            }
        })
        .collect();

    // World-mirror dispatch registration (`pulsar_scenedb::gpu::world_mirror`):
    // a non-generic function with `#name` already concrete, submitted via
    // `inventory` so `World::insert` can find it by `ComponentId` without
    // needing compile-time specialization on the caller's generic `T` --
    // see that module's doc for why the specialization approach doesn't
    // work through a generic `insert_inner<T>` body. Only emitted here (in
    // the non-empty-`gpu_fields` branch) -- a type with no `#[gpu]` fields
    // submits no registration at all, so `World::insert` for it is exactly
    // one `HashMap` miss when a mirror is attached, nothing when it isn't.
    let mirror_dispatch_fn_name = quote::format_ident!("__scenedb_gpu_mirror_dispatch_{}", name);
    let mirror_clear_fn_name = quote::format_ident!("__scenedb_gpu_mirror_clear_{}", name);
    let mirror_descriptors_fn_name =
        quote::format_ident!("__scenedb_gpu_mirror_descriptors_{}", name);
    let mirror_descriptors_fn_default = quote! {
        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #mirror_descriptors_fn_name() -> Vec<::pulsar_scenedb::GpuColumnDesc> {
            <#name #ty_generics as ::pulsar_scenedb::GpuColumnSet>::gpu_columns()
        }
    };

    // Hot World-mutation dispatch is emitted field-by-field. Do not route
    // through `GpuColumnSet::gpu_columns()` here: that reflection API owns a
    // Vec and is appropriate at registration/editor boundaries, not on every
    // `World::insert`. Each DirtyTracked field compares exactly the bytes it
    // would upload against the old component's corresponding field; Once
    // fields keep their presence-lifetime gate and never compare on ordinary
    // updates.
    let mirror_dispatch_fields: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_ident = &f.ident;
            let wrapper = f
                .gpu_wrapper
                .as_ref()
                .expect("gpu field has a wrapper ident");
            match f.mirror_mode {
                MirrorModeAttr::DirtyTracked => quote! {
                    {
                        let value = &data.#field_ident;
                        let bytes: &[u8] =
                            ::pulsar_scenedb::bytemuck::bytes_of(value);
                        let changed = first_handoff || match old_data {
                            Some(old_data) => {
                                let old_value = &old_data.#field_ident;
                                let old_bytes: &[u8] =
                                    ::pulsar_scenedb::bytemuck::bytes_of(old_value);
                                old_bytes != bytes
                            }
                            None => true,
                        };
                        if changed {
                            ::pulsar_scenedb::gpu::world_mirror::write_gpu_column_bytes_at_row(
                                mirror.store(),
                                mirror.queue(),
                                row,
                                ::pulsar_scenedb::component::component_id::<#wrapper>(),
                                ::pulsar_scenedb::MirrorMode::DirtyTracked,
                                bytes,
                                first_handoff,
                            );
                        }
                    }
                },
                MirrorModeAttr::Once => quote! {
                    if first_handoff {
                        let value = &data.#field_ident;
                        let bytes: &[u8] =
                            ::pulsar_scenedb::bytemuck::bytes_of(value);
                        ::pulsar_scenedb::gpu::world_mirror::write_gpu_column_bytes_at_row(
                            mirror.store(),
                            mirror.queue(),
                            row,
                            ::pulsar_scenedb::component::component_id::<#wrapper>(),
                            ::pulsar_scenedb::MirrorMode::Once,
                            bytes,
                            true,
                        );
                    }
                },
            }
        })
        .collect();

    let register_world_owner_calls: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let wrapper = f
                .gpu_wrapper
                .as_ref()
                .expect("gpu field has a wrapper ident");
            quote! {
                store.register_world_gpu_column_owner(
                    owner_component_id,
                    ::pulsar_scenedb::component::component_id::<#wrapper>(),
                );
            }
        })
        .collect();

    let mirror_clear_fields: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let field_ty = &f.ty;
            let wrapper = f
                .gpu_wrapper
                .as_ref()
                .expect("gpu field has a wrapper ident");
            let mirror_mode = match f.mirror_mode {
                MirrorModeAttr::DirtyTracked => {
                    quote! { ::pulsar_scenedb::MirrorMode::DirtyTracked }
                }
                MirrorModeAttr::Once => quote! { ::pulsar_scenedb::MirrorMode::Once },
            };
            quote! {
                {
                    let zero: #field_ty =
                        <#field_ty as ::pulsar_scenedb::bytemuck::Zeroable>::zeroed();
                    ::pulsar_scenedb::gpu::world_mirror::write_gpu_column_bytes_at_row(
                        mirror.store(),
                        mirror.queue(),
                        row,
                        ::pulsar_scenedb::component::component_id::<#wrapper>(),
                        #mirror_mode,
                        ::pulsar_scenedb::bytemuck::bytes_of(&zero),
                        true,
                    );
                }
            }
        })
        .collect();

    // Packed layout (`#[gpu(layout = packed)]`, struct-level): ONE GPU
    // buffer for every `#[gpu]` field combined, instead of the default
    // one-per-field split -- see `struct_is_packed`'s doc for the full
    // rationale and why this is scoped to ONLY `register_gpu_columns_growable`
    // + the World-mirror dispatch fn (`gpu_columns()`/`write_gpu`/the fixed
    // `register_gpu_columns`, generated above, are completely unaffected --
    // still per-field, for the cell-mirrored path, regardless of this flag).
    let packed_view_ident = quote::format_ident!("__ScenedbGpuPacked_{}", name);
    let packed_field_defs: Vec<_> = gpu_fields
        .iter()
        .map(|f| {
            let ident = &f.ident;
            let ty = &f.ty;
            quote! { pub #ident: #ty }
        })
        .collect();
    let packed_field_tys: Vec<_> = gpu_fields.iter().map(|f| &f.ty).collect();
    let packed_field_idents: Vec<_> = gpu_fields.iter().map(|f| f.ident.clone()).collect();
    let packed_view_def = quote! {
        // Field-for-field copy of every #[gpu] field on #name, in
        // declaration order -- NOT a repr(transparent) single-field wrapper
        // like the per-field #[gpu] wrappers above; this is the actual
        // interleaved GPU-side record a shader reads as one struct. Unique
        // by construction (its name embeds #name), so unlike the per-field
        // wrappers it doesn't need the (struct, field) disambiguation
        // trick -- there's exactly one of these per packed struct.
        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        #[repr(C)]
        #[derive(
            Clone,
            Copy,
            ::pulsar_scenedb::bytemuck::Zeroable,
            ::pulsar_scenedb::bytemuck::Pod,
        )]
        pub struct #packed_view_ident {
            #(#packed_field_defs),*
        }
        unsafe impl ::pulsar_scenedb::page::Pod for #packed_view_ident {}
        const _: () = {
            // A shader row is compared and uploaded byte-for-byte. Any
            // implicit repr(C) gap would be uninitialized, so require users
            // to model shader padding as explicit #[gpu] fields instead.
            assert!(
                ::std::mem::size_of::<#packed_view_ident>()
                    == 0usize #(+ ::std::mem::size_of::<#packed_field_tys>())*,
                concat!(
                    "#[gpu(layout = packed)] generated an implicitly padded shader row for ",
                    stringify!(#name),
                    "; add explicit #[gpu] padding fields so every byte is initialized",
                ),
            );
        };
    };

    let register_growable_calls_packed = {
        let buffer_label = format!("{}::packed", name);
        if packed_is_once {
            quote! {
                store.register_once_gpu_buffer::<#packed_view_ident>(initial_capacity, device, #buffer_label);
            }
        } else {
            quote! {
                store.register_dirty_tracked_gpu_buffer::<#packed_view_ident>(initial_capacity, device, #buffer_label);
            }
        }
    };

    let packed_mirror_mode = if packed_is_once {
        quote! { ::pulsar_scenedb::MirrorMode::Once }
    } else {
        quote! { ::pulsar_scenedb::MirrorMode::DirtyTracked }
    };
    let packed_dispatch_body = if packed_is_once {
        quote! {
            if first_handoff {
                let packed = #packed_view_ident {
                    #(#packed_field_idents: data.#packed_field_idents),*
                };
                let bytes: &[u8] = ::pulsar_scenedb::bytemuck::bytes_of(&packed);
                ::pulsar_scenedb::gpu::world_mirror::write_gpu_column_bytes_at_row(
                    mirror.store(),
                    mirror.queue(),
                    row,
                    ::pulsar_scenedb::component::component_id::<#packed_view_ident>(),
                    ::pulsar_scenedb::MirrorMode::Once,
                    bytes,
                    true,
                );
            }
        }
    } else {
        quote! {
            let packed = #packed_view_ident {
                #(#packed_field_idents: data.#packed_field_idents),*
            };
            let bytes: &[u8] = ::pulsar_scenedb::bytemuck::bytes_of(&packed);
            let changed = first_handoff || match old_data {
                Some(old_data) => {
                    let old_packed = #packed_view_ident {
                        #(#packed_field_idents: old_data.#packed_field_idents),*
                    };
                    let old_bytes: &[u8] =
                        ::pulsar_scenedb::bytemuck::bytes_of(&old_packed);
                    old_bytes != bytes
                }
                None => true,
            };
            if changed {
                ::pulsar_scenedb::gpu::world_mirror::write_gpu_column_bytes_at_row(
                    mirror.store(),
                    mirror.queue(),
                    row,
                    ::pulsar_scenedb::component::component_id::<#packed_view_ident>(),
                    ::pulsar_scenedb::MirrorMode::DirtyTracked,
                    bytes,
                    first_handoff,
                );
            }
        }
    };
    let mirror_descriptors_fn_packed = quote! {
        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #mirror_descriptors_fn_name() -> Vec<::pulsar_scenedb::GpuColumnDesc> {
            vec![::pulsar_scenedb::GpuColumnDesc {
                field_token: ::pulsar_scenedb::token::TypeToken::of::<#packed_view_ident>(),
                value_token: ::pulsar_scenedb::token::TypeToken::of::<#packed_view_ident>(),
                field_offset: 0,
                mode: #packed_mirror_mode,
                buffer_name: "packed",
                buffer_key: None,
            }]
        }
    };

    let world_mirror_registration_packed = quote! {
        #packed_view_def
        #mirror_descriptors_fn_packed

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #mirror_dispatch_fn_name(
            mirror: &::pulsar_scenedb::gpu::GpuMirrorHandle,
            row: u32,
            data: *const (),
            old_data: Option<*const ()>,
            first_handoff: bool,
        ) {
            // SAFETY: same contract as the non-packed dispatch fn below --
            // `data` is guaranteed to point at a live, correctly-aligned
            // `#name` (the sole caller, `World::insert_inner`, only reaches
            // this via `#name`'s own `ComponentId`).
            let data = unsafe { &*(data as *const #name #ty_generics) };
            let old_data = old_data.map(|old_data| {
                // SAFETY: `World::insert_inner` supplies this only for an
                // existing value from #name's own archetype column and keeps
                // that value live and unmoved until dispatch returns.
                unsafe { &*(old_data as *const #name #ty_generics) }
            });
            // Assembled via ordinary field access, NOT a raw byte-offset
            // read from `&#name` -- #name's own field layout is compiler-
            // chosen (no repr(C) forced on it) and may interleave #[gpu]
            // and non-#[gpu] fields in any order, so the packed fields are
            // not generally contiguous within #name itself. Building a
            // fresh #packed_view_ident value by name is what makes packed
            // layout correct regardless of #name's actual layout.
            #packed_dispatch_body
        }

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #mirror_clear_fn_name(
            mirror: &::pulsar_scenedb::gpu::GpuMirrorHandle,
            row: u32,
        ) {
            // SAFETY: the generated packed view implements SceneDB Pod, whose
            // contract guarantees that all-zero is a valid value.
            let zero = unsafe { ::std::mem::zeroed::<#packed_view_ident>() };
            let bytes: &[u8] = ::pulsar_scenedb::bytemuck::bytes_of(&zero);
            ::pulsar_scenedb::gpu::world_mirror::write_gpu_column_bytes_at_row(
                mirror.store(),
                mirror.queue(),
                row,
                ::pulsar_scenedb::component::component_id::<#packed_view_ident>(),
                #packed_mirror_mode,
                bytes,
                true,
            );
        }

        ::pulsar_scenedb::pulsar_reflection::inventory::submit! {
            ::pulsar_scenedb::gpu::GpuMirrorRegistration {
                component_id: ::pulsar_scenedb::component::component_id::<#name #ty_generics>,
                descriptors: #mirror_descriptors_fn_name,
                dispatch: #mirror_dispatch_fn_name,
                clear: #mirror_clear_fn_name,
            }
        }
    };

    let world_mirror_registration_default = quote! {
        #mirror_descriptors_fn_default

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #mirror_dispatch_fn_name(
            mirror: &::pulsar_scenedb::gpu::GpuMirrorHandle,
            row: u32,
            data: *const (),
            old_data: Option<*const ()>,
            first_handoff: bool,
        ) {
            // SAFETY: the sole caller, `World::insert_inner`, only reaches
            // this function by looking it up under `#name`'s own
            // `ComponentId` (via `crate::component::component_id::<#name>`,
            // the same key this registration is submitted under below), and
            // passes `&value as *const T as *const ()` for that exact `T` --
            // so `data` is guaranteed to point at a live, correctly-aligned
            // `#name`.
            let data = unsafe { &*(data as *const #name #ty_generics) };
            let old_data = old_data.map(|old_data| {
                // SAFETY: `World::insert_inner` supplies this only for an
                // existing value from #name's own archetype column and keeps
                // that value live and unmoved until dispatch returns.
                unsafe { &*(old_data as *const #name #ty_generics) }
            });
            #(#mirror_dispatch_fields)*
        }

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #mirror_clear_fn_name(
            mirror: &::pulsar_scenedb::gpu::GpuMirrorHandle,
            row: u32,
        ) {
            #(#mirror_clear_fields)*
        }

        ::pulsar_scenedb::pulsar_reflection::inventory::submit! {
            ::pulsar_scenedb::gpu::GpuMirrorRegistration {
                component_id: ::pulsar_scenedb::component::component_id::<#name #ty_generics>,
                descriptors: #mirror_descriptors_fn_name,
                dispatch: #mirror_dispatch_fn_name,
                clear: #mirror_clear_fn_name,
            }
        }
    };

    let (register_growable_calls, world_mirror_registration) = if is_packed {
        (
            vec![register_growable_calls_packed],
            world_mirror_registration_packed,
        )
    } else {
        (register_growable_calls, world_mirror_registration_default)
    };

    let register_growable_descs = if is_packed {
        quote! {
            store.register_gpu_column_descs(#mirror_descriptors_fn_name());
        }
    } else {
        quote! {
            store.register_gpu_column_descs(
                <Self as ::pulsar_scenedb::GpuColumnSet>::gpu_columns()
            );
        }
    };
    let register_world_owners = if is_packed {
        quote! {
            store.register_world_gpu_column_owner(
                owner_component_id,
                ::pulsar_scenedb::component::component_id::<#packed_view_ident>(),
            );
        }
    } else {
        quote! {
            #(#register_world_owner_calls)*
        }
    };

    // Only emitted for packed types: the packed view struct is intentionally
    // unnameable from outside this macro's generated code (same reasoning as
    // the per-field `#[gpu]` wrapper types -- see `FieldInfo::gpu_wrapper`'s
    // doc), so this is the supported way to reach its `ComponentId` (and,
    // through `SceneGpuStore::with_growable_buffer_for_id`/`buffer_for_id`,
    // its actual `wgpu::Buffer`) from outside — needed by anything binding
    // this buffer into a shader (the real motivating use, e.g. Helio).
    let packed_accessor = if is_packed {
        quote! {
            impl #impl_generics #name #ty_generics #where_clause {
                /// `ComponentId` of this type's packed GPU buffer
                /// (registered via [`Self::register_gpu_columns_growable`]).
                pub fn packed_gpu_component_id() -> ::pulsar_scenedb::ComponentId {
                    ::pulsar_scenedb::component::component_id::<#packed_view_ident>()
                }
            }
        }
    } else {
        quote! {}
    };

    quote! {
        unsafe impl #impl_generics ::pulsar_scenedb::GpuColumnSet for #name #ty_generics #where_clause {
            fn gpu_columns() -> Vec<::pulsar_scenedb::GpuColumnDesc> {
                vec![
                    #(#column_descs),*
                ]
            }
            fn write_gpu(
                store: &::pulsar_scenedb::gpu::SceneGpuStore,
                id: ::pulsar_scenedb::gpu::CellId,
                cell: &mut ::pulsar_scenedb::cell::CellStorage,
                handle: ::pulsar_scenedb::handle::Handle,
                data: &Self,
                _phase: &impl ::pulsar_scenedb::gpu::SimulateWitness,
            ) {
                let descs = Self::gpu_columns();
                for desc in &descs {
                    match desc.buffer_name {
                        #(#write_arms)*
                        _ => {}
                    }
                }
            }
        }

        impl #impl_generics #name #ty_generics #where_clause {
            /// Registers this type's `#[gpu]` fields as GPU buffers on
            /// `store` -- one call per field, using the same
            /// disambiguated wrapper types [`Self::write_gpu`] writes
            /// through, so `write_gpu`'s `mark_column_dirty` always finds
            /// a matching buffer instead of silently no-op'ing (the gap
            /// this method exists to close: previously nothing called
            /// `register_gpu_buffer` for derive-generated `#[gpu]` fields
            /// at all).
            ///
            /// Call once per type, at `SceneGpuStore` construction time,
            /// with the same `capacity` every other column on the store
            /// uses (the row-region-partitioned row count -- see
            /// `SceneGpuStore::new`'s own `register_gpu_buffer` calls for
            /// its two built-ins, which this mirrors).
            pub fn register_gpu_columns(
                store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                capacity: u32,
                device: &::wgpu::Device,
            ) {
                store.register_gpu_column_descs(
                    <Self as ::pulsar_scenedb::GpuColumnSet>::gpu_columns()
                );
                #(#register_calls)*
            }

            /// Growable counterpart to [`Self::register_gpu_columns`] --
            /// for World-mirrored use (`World::attach_gpu_mirror`), where
            /// the eventual population of this component isn't known ahead
            /// of time.
            /// `initial_capacity` only needs to be cheap, not sized for the
            /// eventual world; buffers grow transparently on writes past
            /// their current capacity (`SceneGpuStore::write_row_bytes_growing`,
            /// called automatically by `World::insert`'s dispatch path).
            /// Never sets a `max_capacity` ceiling -- see
            /// `SceneGpuStore::register_growable_gpu_buffer`'s doc for why
            /// that's deliberate for World-mirrored columns specifically.
            /// A canonical World buffer has exactly one owning component:
            /// reusing an explicit `buffer = "..."` key from another World
            /// component is rejected because their independent component-
            /// local row allocators cannot safely share physical rows.
            /// Compatible named reuse remains available through fixed
            /// CellStorage registration, whose row regions are disjoint.
            pub fn register_gpu_columns_growable(
                store: &mut ::pulsar_scenedb::gpu::SceneGpuStore,
                initial_capacity: u32,
                device: &::std::sync::Arc<::wgpu::Device>,
            ) {
                #register_growable_descs
                let owner_component_id =
                    ::pulsar_scenedb::component::component_id::<#name #ty_generics>();
                #register_world_owners
                store.register_component_presence_buffer(
                    owner_component_id,
                    initial_capacity,
                    concat!(stringify!(#name), "::presence"),
                );
                #(#register_growable_calls)*
            }
        }

        #packed_accessor

        #world_mirror_registration
    }
}
