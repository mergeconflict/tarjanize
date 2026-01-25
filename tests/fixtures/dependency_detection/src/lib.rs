// =============================================================================
// Test fixture for comprehensive dependency detection testing.
//
// Each section tests a specific type of dependency. The naming convention
// uses prefixes to group related items:
//   - fn_*: function dependency tests
//   - struct_*: struct dependency tests
//   - enum_*: enum dependency tests
//   - trait_*: trait dependency tests
//   - impl_*: impl dependency tests
//   - method_*: method call resolution tests
// =============================================================================

// =============================================================================
// FUNCTION DEPENDENCIES
// =============================================================================

/// Target type for function parameter/return tests.
pub struct FnTargetType;

/// Target function for call tests.
pub fn fn_target() {}

/// Function that calls another function.
/// Expected edge: fn_calls_fn -> fn_target
pub fn fn_calls_fn() {
    fn_target();
}

/// Function with parameter type dependency.
/// Expected edge: fn_param_type -> FnTargetType
pub fn fn_param_type(_x: FnTargetType) {}

/// Function with return type dependency.
/// Expected edge: fn_return_type -> FnTargetType
pub fn fn_return_type() -> FnTargetType {
    FnTargetType
}

/// Function using a type in its body (let binding).
/// Expected edge: fn_body_type -> FnTargetType
pub fn fn_body_type() {
    let _x: FnTargetType = FnTargetType;
}

/// Trait for bound tests.
pub trait FnBoundTrait {}

/// Function with trait bound.
/// Expected edge: fn_trait_bound -> FnBoundTrait
pub fn fn_trait_bound<T: FnBoundTrait>(_x: T) {}

/// Another trait for where clause tests.
pub trait FnWhereTrait {}

/// Function with where clause.
/// Expected edge: fn_where_clause -> FnWhereTrait
pub fn fn_where_clause<T>(_x: T)
where
    T: FnWhereTrait,
{
}

// =============================================================================
// STRUCT DEPENDENCIES
// =============================================================================

/// Target type for struct field tests.
pub struct StructFieldTarget;

/// Struct with a field type dependency.
/// Expected edge: StructWithField -> StructFieldTarget
pub struct StructWithField {
    pub field: StructFieldTarget,
}

/// Trait for struct bound tests.
pub trait StructBoundTrait {}

/// Generic struct with trait bound.
/// Expected edge: StructWithBound -> StructBoundTrait
pub struct StructWithBound<T: StructBoundTrait> {
    pub value: T,
}

/// Trait for struct where clause tests.
pub trait StructWhereTrait {}

/// Generic struct with where clause.
/// Expected edge: StructWithWhere -> StructWhereTrait
pub struct StructWithWhere<T>
where
    T: StructWhereTrait,
{
    pub value: T,
}

// =============================================================================
// ENUM DEPENDENCIES
// =============================================================================

/// Target type for enum variant tests.
pub struct EnumVariantTarget;

/// Enum with tuple variant containing a type.
/// Expected edge: EnumWithTupleVariant -> EnumVariantTarget
pub enum EnumWithTupleVariant {
    Variant(EnumVariantTarget),
}

/// Enum with struct variant containing a type.
/// Expected edge: EnumWithStructVariant -> EnumVariantTarget
pub enum EnumWithStructVariant {
    Variant { field: EnumVariantTarget },
}

/// Trait for enum bound tests.
pub trait EnumBoundTrait {}

/// Generic enum with trait bound.
/// Expected edge: EnumWithBound -> EnumBoundTrait
pub enum EnumWithBound<T: EnumBoundTrait> {
    Variant(T),
}

// =============================================================================
// TRAIT DEPENDENCIES
// =============================================================================

/// Supertrait for inheritance tests.
pub trait Supertrait {}

/// Trait with supertrait dependency.
/// Expected edge: TraitWithSuper -> Supertrait
pub trait TraitWithSuper: Supertrait {}

/// Type used in associated type bounds.
pub struct AssocTypeTarget;

/// Trait for associated type bound tests.
pub trait AssocTypeBound {}

/// Trait with associated type that has a bound.
/// Expected edge: TraitWithAssocTypeBound -> AssocTypeBound
pub trait TraitWithAssocTypeBound {
    type Item: AssocTypeBound;
}

/// Type used in default method body.
pub struct DefaultMethodTarget;

/// Function called from default method.
pub fn default_method_fn_target() {}

/// Trait with default method that has dependencies.
/// Expected edges:
///   - TraitWithDefaultMethod -> DefaultMethodTarget (type in body)
///   - TraitWithDefaultMethod -> default_method_fn_target (function call)
pub trait TraitWithDefaultMethod {
    fn default_method(&self) {
        let _x: DefaultMethodTarget = DefaultMethodTarget;
        default_method_fn_target();
    }
}

/// Type for default const value.
pub struct DefaultConstTarget;

/// Trait with default const that has a type dependency.
/// Expected edge: TraitWithDefaultConst -> DefaultConstTarget
pub trait TraitWithDefaultConst {
    const DEFAULT: Option<DefaultConstTarget> = None;
}

// =============================================================================
// IMPL DEPENDENCIES
// =============================================================================

/// Type for inherent impl tests.
pub struct ImplSelfType;

/// Type used in inherent impl method body.
pub struct InherentImplBodyTarget;

/// Function called from inherent impl method.
pub fn inherent_impl_fn_target() {}

/// Inherent impl with method dependencies.
/// Expected edges (collapsed to impl):
///   - impl ImplSelfType -> ImplSelfType (self type)
///   - impl ImplSelfType -> InherentImplBodyTarget (method body type)
///   - impl ImplSelfType -> inherent_impl_fn_target (method body call)
impl ImplSelfType {
    pub fn method(&self) {
        let _x: InherentImplBodyTarget = InherentImplBodyTarget;
        inherent_impl_fn_target();
    }
}

/// Trait for trait impl tests.
pub trait ImplTrait {
    fn trait_method(&self);
}

/// Type for trait impl tests.
pub struct TraitImplType;

/// Type used in trait impl method body.
pub struct TraitImplBodyTarget;

/// Function called from trait impl method.
pub fn trait_impl_fn_target() {}

/// Trait impl with method dependencies.
/// Expected edges (collapsed to impl):
///   - impl ImplTrait for TraitImplType -> TraitImplType (self type)
///   - impl ImplTrait for TraitImplType -> ImplTrait (trait)
///   - impl ImplTrait for TraitImplType -> TraitImplBodyTarget (method body type)
///   - impl ImplTrait for TraitImplType -> trait_impl_fn_target (method body call)
impl ImplTrait for TraitImplType {
    fn trait_method(&self) {
        let _x: TraitImplBodyTarget = TraitImplBodyTarget;
        trait_impl_fn_target();
    }
}

/// Trait for associated const in impl.
pub trait TraitWithAssocConst {
    const VALUE: i32;
}

/// Type for associated const impl.
pub struct AssocConstImplType;

/// Impl with associated const.
/// Expected edges:
///   - impl TraitWithAssocConst for AssocConstImplType -> AssocConstImplType
///   - impl TraitWithAssocConst for AssocConstImplType -> TraitWithAssocConst
impl TraitWithAssocConst for AssocConstImplType {
    const VALUE: i32 = 42;
}

/// Trait for associated type in impl.
pub trait TraitWithAssocType {
    type Output;
}

/// Type for associated type impl.
pub struct AssocTypeImplType;

/// Target type for associated type.
pub struct AssocTypeOutput;

/// Impl with associated type.
/// Expected edges:
///   - impl TraitWithAssocType for AssocTypeImplType -> AssocTypeImplType
///   - impl TraitWithAssocType for AssocTypeImplType -> TraitWithAssocType
///   - impl TraitWithAssocType for AssocTypeImplType -> AssocTypeOutput
impl TraitWithAssocType for AssocTypeImplType {
    type Output = AssocTypeOutput;
}

// =============================================================================
// METHOD CALL RESOLUTION
// =============================================================================

/// Type with an inherent method for method call tests.
pub struct MethodCallTarget;

impl MethodCallTarget {
    pub fn inherent_method(&self) {}
}

/// Trait with a method for trait method call tests.
pub trait MethodCallTrait {
    fn trait_method(&self);
}

impl MethodCallTrait for MethodCallTarget {
    fn trait_method(&self) {}
}

/// Function that calls an inherent method.
/// Expected edge: method_call_inherent -> impl MethodCallTarget
pub fn method_call_inherent() {
    let x = MethodCallTarget;
    x.inherent_method();
}

/// Function that calls a trait method.
/// Expected edge: method_call_trait -> impl MethodCallTrait for MethodCallTarget
pub fn method_call_trait() {
    let x = MethodCallTarget;
    x.trait_method();
}

// =============================================================================
// CONST AND STATIC DEPENDENCIES
// =============================================================================

/// Type for const tests.
pub struct ConstTypeTarget;

/// Const with type annotation.
/// Expected edge: CONST_WITH_TYPE -> ConstTypeTarget
pub const CONST_WITH_TYPE: Option<ConstTypeTarget> = None;

/// Type for static tests.
pub struct StaticTypeTarget;

/// Static with type annotation.
/// Expected edge: STATIC_WITH_TYPE -> StaticTypeTarget
pub static STATIC_WITH_TYPE: Option<StaticTypeTarget> = None;

// =============================================================================
// TYPE ALIAS DEPENDENCIES
// =============================================================================

/// Target type for type alias tests.
pub struct TypeAliasTarget;

/// Type alias with dependency.
/// Expected edge: AliasOfTarget -> TypeAliasTarget
pub type AliasOfTarget = TypeAliasTarget;

/// Trait for type alias bound tests.
pub trait TypeAliasBoundTrait {}

/// Generic type alias with bound.
/// Expected edge: AliasWithBound -> TypeAliasBoundTrait
pub type AliasWithBound<T: TypeAliasBoundTrait> = Option<T>;

// =============================================================================
// EXTERNAL DEPENDENCY FILTERING
// =============================================================================

/// Function using only std types (should have no local deps).
pub fn fn_uses_std_only() -> String {
    String::new()
}

/// Struct using only std types (should have no local deps).
pub struct StructUsesStdOnly {
    pub value: Vec<i32>,
}
