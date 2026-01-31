//! Test: Associated type constraint creates edges.

pub trait HasItem {
    type Item;
}

pub struct ItemType;

pub fn constrained<T: HasItem<Item = ItemType>>(_: T) {}
