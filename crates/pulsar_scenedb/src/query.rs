use crate::archetype::Archetype;
use crate::component::{component_id, Component, ComponentId};
use crate::entity::Entity;
use crate::world::World;
use std::marker::PhantomData;

/// Types that can be fetched from an archetype row during a query.
///
/// Implementations exist for:
/// - `&T` â€” shared reference to a component
/// - `()` â€” matches every archetype (for counting or iteration without data)
/// - Tuples `(A, B, ...)` up to 8 elements â€” combine multiple fetches
///
/// Mutable component references deliberately belong to [`WorldQueryMut`]
/// and can only be created by [`World::query_mut`]. A shared `&World` must
/// never be enough authority to manufacture `&mut T`.
///
/// # Safety
///
/// Implementations must make [`matches`](Self::matches) describe every
/// column read by [`fetch`](Self::fetch), and `fetch` may only return shared
/// references into the requested row. The access description and fetch must
/// be stable across calls.
pub unsafe trait WorldQuery<'w>: Sized {
    /// The type returned by [`fetch`](Self::fetch).
    type Item;

    /// Returns `true` if the given archetype contains all the components
    /// required by this query.
    fn matches(archetype: &Archetype) -> bool;

    /// Read component data at `row` in `archetype`.
    ///
    /// # Safety
    ///
    /// - `archetype` must satisfy `Self::matches(archetype)`.
    /// - `row` must be < `archetype.entities.len()`.
    unsafe fn fetch(archetype: &'w Archetype, row: usize) -> Self::Item;
}

// SAFETY: this matches and fetches exactly one shared component column.
unsafe impl<'w, T: Component> WorldQuery<'w> for &'w T {
    type Item = &'w T;

    #[inline]
    fn matches(arch: &Archetype) -> bool {
        let cid = component_id::<T>();
        arch.has_columns(std::slice::from_ref(&cid))
    }

    // SAFETY: caller guarantees archetype matches and row is in bounds.
    #[inline]
    unsafe fn fetch(arch: &'w Archetype, row: usize) -> &'w T {
        let cid = component_id::<T>();
        // SAFETY: caller guarantees the column exists and row is in bounds.
        let col = arch.columns.get_unchecked(cid.0 as usize)
            .as_ref()
            .unwrap_unchecked();
        &*(col.get_raw(row) as *const T)
    }
}

// â”€â”€ Empty query: matches every archetype (useful for counting entities) â”€â”€â”€â”€â”€â”€

// SAFETY: the empty query reads no component columns.
unsafe impl<'w> WorldQuery<'w> for () {
    type Item = ();
    #[inline]
    fn matches(_arch: &Archetype) -> bool {
        true
    }
    // SAFETY: caller guarantees row is in bounds.
    #[inline]
    unsafe fn fetch(_arch: &'w Archetype, _row: usize) -> Self::Item {}
}

// â”€â”€ Tuple conbinator macro (1 to 8 components) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

macro_rules! impl_world_query_tuple {
    ($($Q:ident),+) => {
        // SAFETY: every tuple member is itself a valid shared query, so their
        // combination can only return shared references.
        unsafe impl<'w, $($Q: WorldQuery<'w>),+> WorldQuery<'w> for ($($Q,)+) {
            type Item = ($($Q::Item,)+);

            #[inline]
            fn matches(arch: &Archetype) -> bool {
                $($Q::matches(arch))&&+
            }

            // SAFETY: caller guarantees all Q::matches & row in bounds.
            #[inline]
            unsafe fn fetch(arch: &'w Archetype, row: usize) -> Self::Item {
                ($($Q::fetch(arch, row),)+)
            }
        }
    };
}

impl_world_query_tuple!(A);
impl_world_query_tuple!(A, B);
impl_world_query_tuple!(A, B, C);
impl_world_query_tuple!(A, B, C, D);
impl_world_query_tuple!(A, B, C, D, E);
impl_world_query_tuple!(A, B, C, D, E, F);
impl_world_query_tuple!(A, B, C, D, E, F, G);
impl_world_query_tuple!(A, B, C, D, E, F, G, H);

/// Types that can be fetched while holding exclusive access to a [`World`].
///
/// Implementations exist for shared `&T`, exclusive `&mut T`, `()`, and
/// tuples up to 8 elements. Before iteration begins, [`World::query_mut`]
/// validates that no component type is requested mutably more than once, or
/// both mutably and immutably, within the same query item.
///
/// # Safety
///
/// Implementations must enumerate every component reference returned by
/// [`fetch_mut`](Self::fetch_mut), with the correct mutability, in
/// [`for_each_access`](Self::for_each_access). For a matching archetype and
/// in-bounds row, `fetch_mut` must return references only to that row and must
/// obey the access set it reported. The access enumeration must be stable
/// across calls.
pub unsafe trait WorldQueryMut<'w>: Sized {
    type Item;

    fn matches(archetype: &Archetype) -> bool;

    fn for_each_access(f: &mut impl FnMut(ComponentId, bool));

    /// Fetch component data at `row` from an exclusively borrowed world.
    ///
    /// # Safety
    ///
    /// - `archetype` must be valid for the whole `'w` borrow of the world.
    /// - `archetype` must satisfy `Self::matches`.
    /// - `row` must be in bounds.
    /// - the caller must have validated the reported component access set.
    unsafe fn fetch_mut(archetype: *mut Archetype, row: usize) -> Self::Item;
}

// SAFETY: this reports and fetches exactly one shared component column.
unsafe impl<'w, T: Component> WorldQueryMut<'w> for &'w T {
    type Item = &'w T;

    #[inline]
    fn matches(arch: &Archetype) -> bool {
        let cid = component_id::<T>();
        arch.has_columns(std::slice::from_ref(&cid))
    }

    #[inline]
    fn for_each_access(f: &mut impl FnMut(ComponentId, bool)) {
        f(component_id::<T>(), false);
    }

    #[inline]
    unsafe fn fetch_mut(arch: *mut Archetype, row: usize) -> Self::Item {
        // SAFETY: QueryIterMut owns the exclusive World borrow for 'w and
        // validates that this row and column exist before calling us.
        let arch = unsafe { &*arch };
        let cid = component_id::<T>();
        let col = unsafe {
            arch.columns
                .get_unchecked(cid.0 as usize)
                .as_ref()
                .unwrap_unchecked()
        };
        unsafe { &*(col.get_raw(row) as *const T) }
    }
}

// SAFETY: this reports and fetches exactly one exclusive component column.
unsafe impl<'w, T: Component> WorldQueryMut<'w> for &'w mut T {
    type Item = &'w mut T;

    #[inline]
    fn matches(arch: &Archetype) -> bool {
        let cid = component_id::<T>();
        arch.has_columns(std::slice::from_ref(&cid))
    }

    #[inline]
    fn for_each_access(f: &mut impl FnMut(ComponentId, bool)) {
        f(component_id::<T>(), true);
    }

    #[inline]
    unsafe fn fetch_mut(arch: *mut Archetype, row: usize) -> Self::Item {
        // SAFETY: QueryIterMut owns the exclusive World borrow for 'w and
        // rejects any second access that could alias this component column.
        let arch = unsafe { &mut *arch };
        let cid = component_id::<T>();
        let col = unsafe {
            arch.columns
                .get_unchecked_mut(cid.0 as usize)
                .as_mut()
                .unwrap_unchecked()
        };
        unsafe { &mut *(col.get_raw_mut(row) as *mut T) }
    }
}

// SAFETY: the empty query returns no references and reports no accesses.
unsafe impl<'w> WorldQueryMut<'w> for () {
    type Item = ();

    #[inline]
    fn matches(_arch: &Archetype) -> bool {
        true
    }

    #[inline]
    fn for_each_access(_f: &mut impl FnMut(ComponentId, bool)) {}

    #[inline]
    unsafe fn fetch_mut(_arch: *mut Archetype, _row: usize) -> Self::Item {}
}

macro_rules! impl_world_query_mut_tuple {
    ($($Q:ident),+) => {
        // SAFETY: each member reports its complete access set. QueryIterMut
        // validates their union before calling fetch_mut.
        unsafe impl<'w, $($Q: WorldQueryMut<'w>),+> WorldQueryMut<'w> for ($($Q,)+) {
            type Item = ($($Q::Item,)+);

            #[inline]
            fn matches(arch: &Archetype) -> bool {
                $($Q::matches(arch))&&+
            }

            #[inline]
            fn for_each_access(f: &mut impl FnMut(ComponentId, bool)) {
                $($Q::for_each_access(f);)+
            }

            #[inline]
            unsafe fn fetch_mut(arch: *mut Archetype, row: usize) -> Self::Item {
                // SAFETY: the caller validated the combined tuple access set.
                ($(unsafe { $Q::fetch_mut(arch, row) },)+)
            }
        }
    };
}

impl_world_query_mut_tuple!(A);
impl_world_query_mut_tuple!(A, B);
impl_world_query_mut_tuple!(A, B, C);
impl_world_query_mut_tuple!(A, B, C, D);
impl_world_query_mut_tuple!(A, B, C, D, E);
impl_world_query_mut_tuple!(A, B, C, D, E, F);
impl_world_query_mut_tuple!(A, B, C, D, E, F, G);
impl_world_query_mut_tuple!(A, B, C, D, E, F, G, H);

// â”€â”€ QueryIter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Iterator over all entities in the [`World`](crate::World) that match a query
/// pattern `Q`.
///
/// Yields `(Entity, Q::Item)` pairs.  Created by [`World::query`](crate::World::query).
///
/// The iterator scans archetypes in order, skipping those that don't match `Q`.
/// Within each matching archetype it walks rows sequentially.
pub struct QueryIter<'w, Q: WorldQuery<'w>> {
    archetypes: &'w [crate::archetype::Archetype],
    arch_idx: usize,
    row: usize,
    _marker: PhantomData<Q>,
}

impl<'w, Q: WorldQuery<'w>> QueryIter<'w, Q> {
    pub(crate) fn new(world: &'w World) -> Self {
        Self {
            archetypes: &world.archetypes,
            arch_idx: 0,
            row: 0,
            _marker: PhantomData,
        }
    }
}

impl<'w, Q: WorldQuery<'w>> Iterator for QueryIter<'w, Q> {
    type Item = (Entity, Q::Item);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let arch = self.archetypes.get(self.arch_idx)?;
            if !Q::matches(arch) {
                self.arch_idx += 1;
                self.row = 0;
                continue;
            }
            if self.row >= arch.entities.len() {
                self.arch_idx += 1;
                self.row = 0;
                continue;
            }
            let entity = arch.entities[self.row];
            // SAFETY: we've verified that this archetype matches Q and
            // that self.row is in bounds.
            let item = unsafe { Q::fetch(arch, self.row) };
            self.row += 1;
            return Some((entity, item));
        }
    }
}

/// Mutable iterator over all entities matching `Q`.
///
/// Created by [`World::query_mut`]. The raw archetype pointer is an internal
/// implementation detail: `_world` retains the exclusive borrow for `'w`, so
/// the archetype vector cannot be structurally changed while this iterator or
/// any reference yielded by it remains live.
pub struct QueryIterMut<'w, Q: WorldQueryMut<'w>> {
    archetypes: *mut Archetype,
    archetypes_len: usize,
    arch_idx: usize,
    row: usize,
    _world: PhantomData<&'w mut World>,
    _query: PhantomData<Q>,
}

impl<'w, Q: WorldQueryMut<'w>> QueryIterMut<'w, Q> {
    pub(crate) fn new(world: &'w mut World) -> Self {
        validate_mutable_accesses::<Q>();

        #[cfg(feature = "gpu")]
        if world.has_gpu_mirror() {
            Q::for_each_access(&mut |component, mutable| {
                assert!(
                    !mutable
                        || crate::gpu::world_mirror::dispatch_for(component).is_none(),
                    "query_mut cannot borrow GPU-mirrored component {:?} mutably; copy the value and replace it with World::insert so mirror dirty dispatch runs",
                    component,
                );
            });
        }

        Self {
            archetypes: world.archetypes.as_mut_ptr(),
            archetypes_len: world.archetypes.len(),
            arch_idx: 0,
            row: 0,
            _world: PhantomData,
            _query: PhantomData,
        }
    }
}

/// Validate the complete per-item access set without a heap allocation.
/// Query tuples contain at most eight built-in members, but the repeated
/// enumeration also supports external implementations with larger sets.
fn validate_mutable_accesses<'w, Q: WorldQueryMut<'w>>() {
    let mut access_index = 0usize;
    Q::for_each_access(&mut |component, mutable| {
        let this_index = access_index;
        access_index += 1;

        let mut other_index = 0usize;
        Q::for_each_access(&mut |other_component, other_mutable| {
            if other_index < this_index
                && component == other_component
                && (mutable || other_mutable)
            {
                panic!(
                    "query_mut contains aliased access to component {:?}; a component may appear more than once only when every access is shared",
                    component,
                );
            }
            other_index += 1;
        });
    });
}

impl<'w, Q: WorldQueryMut<'w>> Iterator for QueryIterMut<'w, Q> {
    type Item = (Entity, Q::Item);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.arch_idx >= self.archetypes_len {
                return None;
            }

            // SAFETY: the pointer came from the exclusively borrowed World's
            // archetype Vec, which cannot move for the lifetime of this
            // iterator. arch_idx is checked against the captured length.
            let arch = unsafe { self.archetypes.add(self.arch_idx) };
            let arch_ref = unsafe { &*arch };
            if !Q::matches(arch_ref) {
                self.arch_idx += 1;
                self.row = 0;
                continue;
            }
            if self.row >= arch_ref.entities.len() {
                self.arch_idx += 1;
                self.row = 0;
                continue;
            }

            let entity = arch_ref.entities[self.row];
            // SAFETY: construction validated Q's complete access set; the
            // matching archetype and in-bounds row were checked above. Each
            // row is yielded once, and the World stays exclusively borrowed.
            let item = unsafe { Q::fetch_mut(arch, self.row) };
            self.row += 1;
            return Some((entity, item));
        }
    }
}

impl World {
    /// Iterate all entities whose components match the query pattern `Q`.
    ///
    /// # Example
    ///
    /// ```
    /// use pulsar_scenedb::World;
    ///
    /// # struct Pos(f32, f32);
    /// # struct Vel(f32, f32);
    /// # let mut world = World::new();
    /// for (_entity, (pos, vel)) in world.query::<(&Pos, &Vel)>() {
///     // ...
/// }
    /// ```
    ///
    /// An empty tuple `()` matches every archetype and can be used to iterate
    /// all entities without fetching any component data.
    pub fn query<'w, Q: WorldQuery<'w>>(&'w self) -> QueryIter<'w, Q> {
        QueryIter::new(self)
    }

    /// Mutably iterate all entities whose components match `Q`.
    ///
    /// Unlike [`Self::query`], this requires an exclusive World borrow.
    /// Aliasing patterns such as `(&mut T, &mut T)` and `(&T, &mut T)` are
    /// rejected before the first row is visited. When a GPU mirror is
    /// attached, GPU-partnered components are also rejected: an unstructured
    /// `&mut T` cannot publish field-level dirty state, so those values must be
    /// copied, edited, and replaced through [`World::insert`].
    ///
    /// ```
    /// use pulsar_scenedb::World;
    ///
    /// # #[derive(Debug)]
    /// # struct Pos(f32);
    /// # struct Vel(f32);
    /// # let mut world = World::new();
    /// # let entity = world.spawn();
    /// # world.insert(entity, Pos(1.0));
    /// # world.insert(entity, Vel(2.0));
    /// for (_entity, (pos, vel)) in world.query_mut::<(&mut Pos, &Vel)>() {
    ///     pos.0 += vel.0;
    /// }
    /// ```
    pub fn query_mut<'w, Q: WorldQueryMut<'w>>(&'w mut self) -> QueryIterMut<'w, Q> {
        QueryIterMut::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Pos(i32);
    struct Vel(i32);

    #[test]
    fn mixed_mutable_and_shared_query_updates_distinct_columns() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Pos(3));
        world.insert(entity, Vel(4));

        for (_, (pos, vel)) in world.query_mut::<(&mut Pos, &Vel)>() {
            pos.0 += vel.0;
        }

        assert_eq!(world.get::<Pos>(entity), Some(&Pos(7)));
    }

    #[test]
    #[should_panic(expected = "aliased access")]
    fn duplicate_mutable_access_is_rejected_before_iteration() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Pos(1));
        let _ = world.query_mut::<(&mut Pos, &mut Pos)>();
    }

    #[test]
    #[should_panic(expected = "aliased access")]
    fn shared_and_mutable_access_to_same_component_is_rejected() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Pos(1));
        let _ = world.query_mut::<(&Pos, &mut Pos)>();
    }
}
