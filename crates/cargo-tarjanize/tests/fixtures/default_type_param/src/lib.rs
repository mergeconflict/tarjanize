//! Test: Default type parameter creates edge.

pub struct Default;

pub struct WithDefault<T = Default>(T);
