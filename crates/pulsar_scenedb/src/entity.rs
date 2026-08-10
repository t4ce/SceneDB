use std::fmt;

/// Entity-slot row sentinel used while a slot is free or permanently
/// retired. Unlike an archetype id, a row can represent emptiness without
/// colliding with the valid empty archetype (which contains live entities).
pub(crate) const DEAD_ROW: u32 = u32::MAX;

/// A lightweight handle to an entity in the ECS [`World`](crate::World).
///
/// Internally a packed `u64`: the lower 32 bits are the entity index (slot in
/// [`World::entity_slots`]) and the upper 32 bits are the generation counter.
/// The generation is advanced when a live slot is despawned, before it enters
/// the free list. Generation `u32::MAX` permanently retires the slot instead
/// of wrapping, which lets [`World::is_alive`] reject stale handles forever.
///
/// `Entity` is `Copy`, cheap to pass around, and `DANGLING` can be used as
/// a sentinel value.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Entity(u64);

impl Entity {
    #[inline]
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self(((generation as u64) << 32) | (index as u64))
    }

    /// The slot index within [`World::entity_slots`](crate::World).
    #[inline]
    pub fn index(self) -> u32 {
        self.0 as u32
    }

    /// The generation counter for stale-handle detection.
    ///
    /// Each despawn advances the slot generation. A handle is alive only when
    /// both its generation matches and the slot is currently occupied; a free
    /// slot never becomes live merely because forged bits match its next
    /// generation.
    #[inline]
    pub fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Sentinel value for a dead or null entity.
    ///
    /// `u64::MAX` — guaranteed not to collide with any valid entity because
    /// generation `u32::MAX` is reserved for permanent retirement. Useful as
    /// an initializer for option-like patterns without heap allocation.
    pub const DANGLING: Entity = Entity(u64::MAX);

    /// Raw packed u64 representation (for serialization).
    #[inline]
    pub fn bits(self) -> u64 {
        self.0
    }

    /// Construct an Entity from its raw packed u64 representation.
    #[inline]
    pub fn from_bits(bits: u64) -> Self {
        Self(bits)
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.index(), self.generation())
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EntitySlot {
    pub generation: u32,
    pub archetype: crate::archetype::ArchetypeId,
    pub row: u32,
}

impl EntitySlot {
    pub(crate) fn empty(generation: u32) -> Self {
        Self {
            generation,
            archetype: crate::archetype::ArchetypeId::EMPTY,
            row: DEAD_ROW,
        }
    }
}
