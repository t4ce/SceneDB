use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::scene_store::FieldInfo;

/// `use_gpu_wrappers` picks which type each `#[gpu]` field's CellType
/// column token is declared for: the field's own generated wrapper type
/// for a GPU-bearing component (which must match what `write_gpu` and
/// `GpuColumnDesc` use; see `FieldInfo::gpu_wrapper`), or the field's raw
/// type for a CPU-only component. `scene_store::expand` selects this from
/// the component's actual fields, never from an ambient feature in the
/// consuming crate.
pub fn generate_scene_column_set(
    name: &Ident,
    impl_generics: &syn::ImplGenerics,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&syn::WhereClause>,
    field_infos: &[FieldInfo],
    use_gpu_wrappers: bool,
) -> TokenStream {
    let cell_type_name = name.to_string();

    let cell_entries: Vec<_> = field_infos
        .iter()
        .map(|f| {
            if use_gpu_wrappers {
                if let Some(wrapper) = &f.gpu_wrapper {
                    return quote! {
                        .with(::pulsar_scenedb::token::TypeToken::of::<#wrapper>())
                    };
                }
            }
            let ty = &f.ty;
            quote! {
                .with(::pulsar_scenedb::token::TypeToken::of::<#ty>())
            }
        })
        .collect();

    quote! {
        impl #impl_generics ::pulsar_scenedb::cell_type::SceneColumnSet for #name #ty_generics #where_clause {
            fn cell_type() -> ::pulsar_scenedb::cell_type::RegisteredCellType {
                ::pulsar_scenedb::cell_type::CellType::new(#cell_type_name)
                    #(#cell_entries)*
                    .build()
                    .expect("SceneColumnSet cell_type: CellType::build failed (duplicate columns or stride overflow)")
            }
        }
    }
}
