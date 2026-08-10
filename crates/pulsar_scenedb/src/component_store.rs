use pulsar_reflection::{EngineClass, REGISTRY, RUNTIME_TYPE_REGISTRY};
use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::ptr;
use std::ptr::NonNull;

thread_local! {
    static BP_COMP_CTX: RefCell<Vec<NonNull<ComponentStore>>> = const {
        RefCell::new(Vec::new())
    };
}

/// Run `f` with `store` installed as the thread-local blueprint component
/// context.
///
/// This is the safe executor entry point. The context is scoped to the call,
/// is restored during unwinding, and keeps the exclusive borrow of `store`
/// alive for the whole callback. Nested scopes are supported; a nested
/// [`__bp_with_comp`] access while another access callback is still running
/// is rejected rather than creating aliased mutable references.
#[inline]
pub fn __bp_with_comp_ctx<R>(store: &mut ComponentStore, f: impl FnOnce() -> R) -> R {
    struct ContextGuard(NonNull<ComponentStore>);

    impl Drop for ContextGuard {
        fn drop(&mut self) {
            BP_COMP_CTX.with(|contexts| {
                let popped = contexts
                    .borrow_mut()
                    .pop()
                    .expect("unbalanced blueprint component context stack");
                assert_eq!(
                    popped, self.0,
                    "blueprint component contexts must unwind in LIFO order"
                );
            });
        }
    }

    let ptr = NonNull::from(store);
    BP_COMP_CTX.with(|contexts| contexts.borrow_mut().push(ptr));
    let _guard = ContextGuard(ptr);
    f()
}

/// Install a thread-local blueprint component context without a scoped
/// lifetime guard.
///
/// Prefer [`__bp_with_comp_ctx`]. This compatibility hook exists for
/// executors whose ABI still requires separate enter/leave calls.
///
/// # Safety
///
/// Until a matching [`__bp_clear_comp_ctx`] on the same thread, `store` must
/// remain alive and must not be accessed through any path other than
/// [`__bp_with_comp`]. Calls must be balanced in LIFO order, including during
/// unwinding.
#[inline]
pub unsafe fn __bp_set_comp_ctx(store: &mut ComponentStore) {
    BP_COMP_CTX.with(|contexts| contexts.borrow_mut().push(NonNull::from(store)));
}

/// Remove a context installed by [`__bp_set_comp_ctx`].
///
/// Prefer [`__bp_with_comp_ctx`], which cannot leak a pointer when a callback
/// panics.
///
/// # Safety
///
/// This must pair with the most recent unmatched [`__bp_set_comp_ctx`] call
/// on the same thread, after every access callback has returned.
#[inline]
pub unsafe fn __bp_clear_comp_ctx() {
    BP_COMP_CTX.with(|contexts| {
        contexts
            .borrow_mut()
            .pop()
            .expect("blueprint component context stack is empty");
    });
}

/// Access the current blueprint component store from thread-local context.
///
/// # Panics
///
/// Panics if called outside [`__bp_with_comp_ctx`] (or an unsafe legacy
/// `__bp_set_comp_ctx` / `__bp_clear_comp_ctx` pair), or if called recursively
/// while another access callback is still borrowing the store.
#[inline]
pub fn __bp_with_comp<R>(f: impl FnOnce(&mut ComponentStore) -> R) -> R {
    BP_COMP_CTX.with(|contexts| {
        // Keep the dynamic borrow alive across `f`: a recursive accessor (or
        // an unsafe enter/leave call from inside `f`) then panics before it can
        // manufacture a second `&mut ComponentStore` to the same allocation.
        let mut contexts = contexts.try_borrow_mut().unwrap_or_else(|_| {
            panic!("recursive blueprint component access would alias a mutable store")
        });
        let ptr = contexts
            .last_mut()
            .expect("Blueprint component access outside Actor lifecycle");
        // SAFETY: the safe scoped entry point retains the exclusive borrow for
        // the callback lifetime. The legacy entry point requires the same
        // invariant from its unsafe caller. The RefCell borrow above prevents
        // reentrant safe access from creating another mutable reference.
        unsafe { f(ptr.as_mut()) }
    })
}

/// Runtime store for blueprint (visual scripting) components attached to an
/// actor or object.
///
/// Each entry is a `(class_name, Box<dyn EngineClass>)` pair.  The
/// `EngineClass` trait comes from `pulsar_reflection` and provides
/// reflection-based property get/set and method dispatch via JSON.
///
/// This is the bridge between the ECS world and the blueprint runtime:
/// blueprint instances read and write their reflected properties through
/// a `ComponentStore` rather than through direct ECS column access.
///
/// [`__bp_with_comp_ctx`] and [`__bp_with_comp`] let blueprint VM bytecode
/// operate on the *current* actor's store without plumbing it through every
/// call site. Separate enter/leave hooks remain available only as unsafe
/// compatibility APIs because a raw thread-local pointer cannot carry the
/// source borrow's lifetime.
pub struct ComponentStore {
    entries: Vec<(String, Box<dyn EngineClass>)>,
}

impl Default for ComponentStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentStore {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create a component from the reflection registry and deserialize its
    /// properties from a JSON map.
    ///
    /// Returns `false` if `class_name` is not registered in the reflection
    /// registry.
    pub fn add_from_registry(&mut self, class_name: &str, data: &serde_json::Value) -> bool {
        let Some(mut instance) = REGISTRY.create_instance(class_name) else {
            tracing::warn!(
                "ComponentStore: unknown class '{}' â€” not in reflection registry",
                class_name
            );
            return false;
        };

        if let Some(obj) = data.as_object() {
            let apply_list: Vec<_> = {
                let props = instance.get_properties();
                props
                    .into_iter()
                    .filter_map(|prop| {
                        obj.get(prop.name)
                            .cloned()
                            .map(|jv| (prop.type_info, prop.setter, jv))
                    })
                    .collect()
            };

            for (type_info, setter, json_val) in apply_list {
                match RUNTIME_TYPE_REGISTRY.deserialize_json_for_type(type_info, json_val) {
                    Ok(any_val) => (setter)(instance.as_mut(), any_val),
                    Err(e) => {
                        tracing::warn!(
                            "ComponentStore: failed to apply property on '{}': {}",
                            class_name,
                            e
                        );
                    }
                }
            }
        }

        self.entries.push((class_name.to_string(), instance));
        true
    }

    /// Add a pre-constructed engine-class instance.
    pub fn add_boxed(&mut self, class_name: impl Into<String>, comp: Box<dyn EngineClass>) {
        self.entries.push((class_name.into(), comp));
    }

    /// Get a shared reference to the first component of type `T`.
    pub fn get<T: EngineClass + 'static>(&self) -> Option<&T> {
        self.entries
            .iter()
            .find_map(|(_, e)| e.as_any().downcast_ref::<T>())
    }

    /// Get a mutable reference to the first component of type `T`.
    pub fn get_mut<T: EngineClass + 'static>(&mut self) -> Option<&mut T> {
        self.entries
            .iter_mut()
            .find_map(|(_, e)| e.as_any_mut().downcast_mut::<T>())
    }

    /// Get a shared reference to a component by its registered class name.
    pub fn get_by_name(&self, class_name: &str) -> Option<&dyn EngineClass> {
        self.entries
            .iter()
            .find(|(name, _)| name == class_name)
            .map(|(_, e)| e.as_ref())
    }

    /// Get a mutable reference to a component by its registered class name.
    pub fn get_by_name_mut(&mut self, class_name: &str) -> Option<&mut dyn EngineClass> {
        self.entries
            .iter_mut()
            .find(|(name, _)| name == class_name)
            .map(|(_, e)| e.as_mut())
    }

    /// Serialize a component property to JSON.
    pub fn get_property_json(
        &self,
        class_name: &str,
        prop_name: &str,
    ) -> Option<serde_json::Value> {
        let (_, comp) = self.entries.iter().find(|(name, _)| name == class_name)?;

        let props = comp.get_properties();
        let prop = props.into_iter().find(|p| p.name == prop_name)?;
        let any_val: Box<dyn std::any::Any> = (prop.getter)(comp.as_ref());
        RUNTIME_TYPE_REGISTRY
            .serialize_json_for_any(any_val.as_ref())
            .ok()
    }

    /// Deserialize a JSON value into a component property.
    ///
    /// Returns `false` if `class_name` or `prop_name` is not found.
    pub fn set_property_json(
        &mut self,
        class_name: &str,
        prop_name: &str,
        value: serde_json::Value,
    ) -> bool {
        let Some(idx) = self.entries.iter().position(|(name, _)| name == class_name) else {
            return false;
        };

        let (type_info, setter) = {
            let comp_ref = self.entries[idx].1.as_ref();
            let props = comp_ref.get_properties();
            match props.into_iter().find(|p| p.name == prop_name) {
                Some(prop) => (prop.type_info, prop.setter),
                None => return false,
            }
        };

        let any_val = match RUNTIME_TYPE_REGISTRY.deserialize_json_for_type(type_info, value) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "ComponentStore::set_property_json failed for {}.{}: {}",
                    class_name,
                    prop_name,
                    e
                );
                return false;
            }
        };

        let comp_mut = self.entries[idx].1.as_mut();
        (setter)(comp_mut, any_val);
        true
    }

    /// Directly write raw bytes into a reflected component property.
    ///
    /// Reconstructs a `Box<dyn Any>` from `ptr`/`size` and dispatches
    /// through the reflection system's setter.  This avoids JSON
    /// serialization overhead on the hot blueprint VM path while keeping
    /// the existing setter logic (type validation, side-effects, etc.)
    /// intact.
    ///
    /// Only exact, known Copy primitive identities are accepted; equal byte
    /// size is not a type identity (`u32` and `f32` must never be confused).
    /// Arrays used by the reflection primitives (`[f32; 2/3/4/16]`) are
    /// accepted as well. Other compound/custom types return `false` without
    /// modifying the property.
    /// The blueprint VM should fall back to [`set_property_json`] for
    /// compound types.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to a valid, properly-aligned block of `size` bytes.
    /// - The bytes at `ptr` represent a valid instance of the target
    ///   property's type (same layout, size, and alignment).
    /// - `size` matches the exact size of the property's type, as
    ///   returned by the reflection type info.
    pub unsafe fn set_property_raw(
        &mut self,
        class_name: &str,
        prop_name: &str,
        ptr: *const u8,
        size: usize,
    ) -> bool {
        let Some(idx) = self.entries.iter().position(|(name, _)| name == class_name) else {
            return false;
        };

        let (setter, type_info) = {
            let comp_ref = self.entries[idx].1.as_ref();
            let props = comp_ref.get_properties();
            match props.into_iter().find(|p| p.name == prop_name) {
                Some(prop) => (prop.setter, prop.type_info),
                None => return false,
            }
        };

        if ptr.is_null()
            || size != type_info.size
            || (ptr as usize) % type_info.align.max(1) != 0
        {
            return false;
        }

        macro_rules! read_copy {
            ($ty:ty) => {{
                // SAFETY: the method's caller guarantees a live value of the
                // reflected property type; the TypeId/size/alignment checks
                // above prove this arm reads that exact Copy type.
                Box::new(unsafe { ptr::read(ptr.cast::<$ty>()) }) as Box<dyn Any>
            }};
        }

        let type_id = type_info.type_id;
        let any_val: Box<dyn Any> = if type_id == TypeId::of::<bool>() {
            read_copy!(bool)
        } else if type_id == TypeId::of::<u8>() {
            read_copy!(u8)
        } else if type_id == TypeId::of::<i8>() {
            read_copy!(i8)
        } else if type_id == TypeId::of::<u16>() {
            read_copy!(u16)
        } else if type_id == TypeId::of::<i16>() {
            read_copy!(i16)
        } else if type_id == TypeId::of::<u32>() {
            read_copy!(u32)
        } else if type_id == TypeId::of::<i32>() {
            read_copy!(i32)
        } else if type_id == TypeId::of::<u64>() {
            read_copy!(u64)
        } else if type_id == TypeId::of::<i64>() {
            read_copy!(i64)
        } else if type_id == TypeId::of::<u128>() {
            read_copy!(u128)
        } else if type_id == TypeId::of::<i128>() {
            read_copy!(i128)
        } else if type_id == TypeId::of::<usize>() {
            read_copy!(usize)
        } else if type_id == TypeId::of::<isize>() {
            read_copy!(isize)
        } else if type_id == TypeId::of::<f32>() {
            read_copy!(f32)
        } else if type_id == TypeId::of::<f64>() {
            read_copy!(f64)
        } else if type_id == TypeId::of::<char>() {
            read_copy!(char)
        } else if type_id == TypeId::of::<[f32; 2]>() {
            read_copy!([f32; 2])
        } else if type_id == TypeId::of::<[f32; 3]>() {
            read_copy!([f32; 3])
        } else if type_id == TypeId::of::<[f32; 4]>() {
            read_copy!([f32; 4])
        } else if type_id == TypeId::of::<[f32; 16]>() {
            read_copy!([f32; 16])
        } else {
            return false;
        };

        let comp_mut = self.entries[idx].1.as_mut();
        (setter)(comp_mut, any_val);
        true
    }

    /// Call a reflected method on a component with JSON-serialized arguments.
    ///
    /// Returns the JSON-serialized return value, if any.
    pub fn call_method_json(
        &mut self,
        class_name: &str,
        method_name: &str,
        args: Vec<serde_json::Value>,
    ) -> Option<serde_json::Value> {
        let methods = REGISTRY.get_methods(class_name)?;
        let method = methods.into_iter().find(|m| m.name == method_name)?;

        let idx = self
            .entries
            .iter()
            .position(|(name, _)| name == class_name)?;

        let mut any_args: Vec<Box<dyn std::any::Any>> = Vec::new();
        for (param, json_val) in method.params.iter().zip(args.into_iter()) {
            match RUNTIME_TYPE_REGISTRY.deserialize_json_for_type(param.type_info, json_val) {
                Ok(v) => any_args.push(v),
                Err(e) => {
                    tracing::warn!("ComponentStore::call_method_json arg error: {}", e);
                    return None;
                }
            }
        }

        let comp_mut = self.entries[idx].1.as_mut();
        let result = (method.caller)(comp_mut, any_args);

        result.and_then(|rv| {
            RUNTIME_TYPE_REGISTRY
                .serialize_json_for_any(rv.as_ref())
                .ok()
        })
    }

    /// Returns `true` if a component with `class_name` is stored.
    pub fn has(&self, class_name: &str) -> bool {
        self.entries.iter().any(|(name, _)| name == class_name)
    }

    /// Number of component entries in this store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(class_name, component)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn EngineClass)> {
        self.entries.iter().map(|(n, e)| (n.as_str(), e.as_ref()))
    }

    /// Iterate mutably over `(class_name, component)` pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut dyn EngineClass)> {
        self.entries
            .iter_mut()
            .map(|(n, e)| (n.as_str(), e.as_mut()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pulsar_reflection::{PropertyMetadata, Reflectable};

    #[derive(Clone, Default)]
    struct RawPropertyFixture {
        integer: i32,
        scalar: f32,
    }

    fn property<T: Copy + Any + Send + Sync + Reflectable + 'static>(
        name: &'static str,
        get: fn(&RawPropertyFixture) -> T,
        set: fn(&mut RawPropertyFixture, T),
    ) -> PropertyMetadata {
        PropertyMetadata {
            name,
            display_name: name.to_owned(),
            category: None,
            category_color: None,
            category_default_collapsed: false,
            category_order: None,
            type_info: T::type_info(),
            getter: Box::new(move |component| {
                let component = component
                    .as_any()
                    .downcast_ref::<RawPropertyFixture>()
                    .expect("fixture getter type");
                Box::new(get(component))
            }),
            setter: Box::new(move |component, value| {
                let component = component
                    .as_any_mut()
                    .downcast_mut::<RawPropertyFixture>()
                    .expect("fixture setter component type");
                let value = *value.downcast::<T>().expect("fixture setter value type");
                set(component, value);
            }),
        }
    }

    impl EngineClass for RawPropertyFixture {
        fn class_name() -> &'static str {
            "RawPropertyFixture"
        }

        fn get_properties(&self) -> Vec<PropertyMetadata> {
            vec![
                property("integer", |fixture| fixture.integer, |fixture, value| {
                    fixture.integer = value;
                }),
                property("scalar", |fixture| fixture.scalar, |fixture, value| {
                    fixture.scalar = value;
                }),
            ]
        }

        fn create_default() -> Box<dyn EngineClass> {
            Box::<Self>::default()
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn clone_boxed(&self) -> Box<dyn EngineClass> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn raw_property_write_uses_reflected_type_identity_not_equal_size() {
        let mut store = ComponentStore::new();
        store.add_boxed(
            RawPropertyFixture::class_name(),
            Box::<RawPropertyFixture>::default(),
        );

        let integer = -1_234_567_i32;
        let scalar = 37.25_f32;
        // SAFETY: both pointers name live, aligned values of the exact
        // reflected property type for the duration of each call.
        unsafe {
            assert!(store.set_property_raw(
                RawPropertyFixture::class_name(),
                "integer",
                (&integer as *const i32).cast(),
                std::mem::size_of::<i32>(),
            ));
            assert!(store.set_property_raw(
                RawPropertyFixture::class_name(),
                "scalar",
                (&scalar as *const f32).cast(),
                std::mem::size_of::<f32>(),
            ));
        }

        let fixture = store
            .get::<RawPropertyFixture>()
            .expect("fixture remains stored");
        assert_eq!(fixture.integer, integer);
        assert_eq!(fixture.scalar.to_bits(), scalar.to_bits());
    }

    #[test]
    fn scoped_blueprint_context_restores_after_unwind() {
        let mut store = ComponentStore::new();
        store.add_boxed(
            RawPropertyFixture::class_name(),
            Box::<RawPropertyFixture>::default(),
        );

        __bp_with_comp_ctx(&mut store, || {
            __bp_with_comp(|current| assert_eq!(current.len(), 1));
        });

        let outside = std::panic::catch_unwind(|| __bp_with_comp(|_| ()));
        assert!(outside.is_err(), "context pointer escaped its scoped borrow");

        let callback_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            __bp_with_comp_ctx(&mut store, || panic!("fixture panic"));
        }));
        assert!(callback_panic.is_err());
        let outside = std::panic::catch_unwind(|| __bp_with_comp(|_| ()));
        assert!(outside.is_err(), "panicking callback leaked its context");
    }

    #[test]
    fn blueprint_context_rejects_recursive_mutable_access() {
        let mut store = ComponentStore::new();
        __bp_with_comp_ctx(&mut store, || {
            let recursive = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                __bp_with_comp(|_| {
                    __bp_with_comp(|_| ());
                });
            }));
            assert!(recursive.is_err());

            // The failed nested borrow must not poison the active context.
            __bp_with_comp(|current| assert!(current.is_empty()));
        });
    }
}
