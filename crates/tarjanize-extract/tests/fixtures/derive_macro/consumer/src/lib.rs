use my_derive::MyDerive;

/// A struct using the workspace-local derive macro.
///
/// If derive macros are captured, `Foo` should have a dependency on `MyDerive`.
#[derive(MyDerive)]
pub struct Foo;
