# ra_ap_syntax

Syntax Tree library used throughout rust-analyzer. Provides both a Concrete Syntax Tree (CST) and an Abstract Syntax Tree (AST) built on top of it.

## Overview

**Properties:**
- Easy and fast incremental re-parsing
- Graceful handling of errors (always produces a tree, even for invalid code)
- Full-fidelity representation (any text can be precisely represented)

The design is inspired by the [Swift libSyntax](https://github.com/apple/swift/blob/13d593df6f359d0cb2fc81cfaac273297c539455/lib/Syntax/README.md) and documented in [RFC 2256](https://github.com/rust-lang/rfcs/pull/2256).

**Architecture:**
- CST is defined in `syntax_node` module (via `rowan`)
- AST is layered on top in `ast` module
- The actual parser lives in a separate `parser` crate; the lexer lives here

## Re-exports

```rust
// Core types from this crate
pub use crate::{
    ast::{AstNode, AstToken},
    ptr::{AstPtr, SyntaxNodePtr},
    syntax_error::SyntaxError,
    syntax_node::{
        PreorderWithTokens, RustLanguage, SyntaxElement, SyntaxElementChildren,
        SyntaxNode, SyntaxNodeChildren, SyntaxToken, SyntaxTreeBuilder,
    },
    token_text::TokenText,
};

// From parser crate
pub use parser::{Edition, SyntaxKind, T};

// From rowan crate
pub use rowan::{
    Direction, GreenNode, NodeOrToken, SyntaxText, TextRange, TextSize,
    TokenAtOffset, WalkEvent, api::Preorder,
};

// String types
pub use smol_str::{SmolStr, SmolStrBuilder, ToSmolStr, format_smolstr};

// Literal unescaping
pub use rustc_literal_escaper as unescape;
```

---

## Parse<T>

The result of parsing: a syntax tree and a collection of errors. Always produces a tree, even for completely invalid files.

```rust
pub struct Parse<T> {
    green: Option<GreenNode>,
    errors: Option<Arc<[SyntaxError]>>,
    _ty: PhantomData<fn() -> T>,
}
```

### Methods (all `Parse<T>`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `syntax_node` | `fn syntax_node(&self) -> SyntaxNode` | Get the untyped syntax tree root |
| `errors` | `fn errors(&self) -> Vec<SyntaxError>` | Get parse and validation errors |

### Methods (`Parse<T: AstNode>`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_syntax` | `fn to_syntax(self) -> Parse<SyntaxNode>` | Convert to untyped parse result |
| `tree` | `fn tree(&self) -> T` | Get typed AST node (panics if root is ERROR) |
| `ok` | `fn ok(self) -> Result<T, Vec<SyntaxError>>` | Convert to Result based on errors |

### Methods (`Parse<SyntaxNode>`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `cast` | `fn cast<N: AstNode>(self) -> Option<Parse<N>>` | Try to cast to typed parse result |

### Methods (`Parse<SourceFile>`)

| Method | Signature | Description |
|--------|-----------|-------------|
| `debug_dump` | `fn debug_dump(&self) -> String` | Debug output with tree and errors |
| `reparse` | `fn reparse(&self, delete: TextRange, insert: &str, edition: Edition) -> Parse<SourceFile>` | Incremental reparse |

---

## SourceFile

The entry point for parsing. Represents a parse tree for a single Rust file.

```rust
impl SourceFile {
    pub fn parse(text: &str, edition: Edition) -> Parse<SourceFile>;
}
```

### Parsing Expressions

```rust
impl ast::Expr {
    /// Parse an expression from text.
    /// Panics in tree() if the result isn't a valid expression.
    pub fn parse(text: &str, edition: Edition) -> Parse<ast::Expr>;
}
```

---

## Concrete Syntax Tree (CST)

The CST includes comments and whitespace, providing full-fidelity representation. Built on the `rowan` crate.

### RustLanguage

The language marker type for rowan:

```rust
pub enum RustLanguage {}

impl Language for RustLanguage {
    type Kind = SyntaxKind;
    // ...
}
```

### Core Types (rowan wrappers)

| Type | Description |
|------|-------------|
| `SyntaxNode` | `rowan::SyntaxNode<RustLanguage>` - A node in the tree |
| `SyntaxToken` | `rowan::SyntaxToken<RustLanguage>` - A leaf token |
| `SyntaxElement` | `rowan::SyntaxElement<RustLanguage>` - Either node or token |
| `SyntaxNodeChildren` | Iterator over child nodes |
| `SyntaxElementChildren` | Iterator over child elements |
| `PreorderWithTokens` | Preorder traversal including tokens |

### SyntaxNode Key Methods

| Method | Description |
|--------|-------------|
| `kind()` | Get the `SyntaxKind` |
| `text_range()` | Get the `TextRange` in source |
| `text()` | Get the `SyntaxText` (lazy concatenation of tokens) |
| `parent()` | Get parent node |
| `children()` | Iterate over child nodes |
| `children_with_tokens()` | Iterate over child elements |
| `first_child()` / `last_child()` | Get first/last child node |
| `first_token()` / `last_token()` | Get first/last token |
| `next_sibling()` / `prev_sibling()` | Navigate siblings |
| `ancestors()` | Iterate up the tree |
| `descendants()` | Iterate down the tree (preorder) |
| `descendants_with_tokens()` | Include tokens in descent |
| `preorder()` | Preorder traversal with enter/leave events |
| `preorder_with_tokens()` | Include tokens |
| `siblings(direction)` | Iterate siblings in a direction |
| `covering_element(range)` | Find smallest element covering a range |
| `token_at_offset(offset)` | Find token(s) at offset |
| `clone_for_update()` | Clone for mutable editing |
| `clone_subtree()` | Clone just this subtree |

### SyntaxTreeBuilder

Build syntax trees programmatically:

```rust
impl SyntaxTreeBuilder {
    pub fn token(&mut self, kind: SyntaxKind, text: &str);
    pub fn start_node(&mut self, kind: SyntaxKind);
    pub fn finish_node(&mut self);
    pub fn error(&mut self, error: String, text_pos: TextSize);
    pub fn finish(self) -> Parse<SyntaxNode>;
}
```

---

## Abstract Syntax Tree (AST)

The AST is layered on top of the CST, providing typed access to syntax nodes. The conversion has **zero runtime cost** - AST nodes are just newtype wrappers around `SyntaxNode`.

### AstNode Trait

The main trait for typed AST nodes:

```rust
pub trait AstNode {
    /// Static kind (panics if dynamic)
    fn kind() -> SyntaxKind where Self: Sized;

    /// Check if a SyntaxKind can be cast to this type
    fn can_cast(kind: SyntaxKind) -> bool where Self: Sized;

    /// Try to cast a SyntaxNode to this type
    fn cast(syntax: SyntaxNode) -> Option<Self> where Self: Sized;

    /// Get the underlying SyntaxNode
    fn syntax(&self) -> &SyntaxNode;

    /// Clone for mutable editing
    fn clone_for_update(&self) -> Self where Self: Sized;

    /// Clone just the subtree
    fn clone_subtree(&self) -> Self where Self: Sized;
}
```

### AstToken Trait

For typed access to tokens:

```rust
pub trait AstToken {
    fn can_cast(token: SyntaxKind) -> bool where Self: Sized;
    fn cast(syntax: SyntaxToken) -> Option<Self> where Self: Sized;
    fn syntax(&self) -> &SyntaxToken;
    fn text(&self) -> &str;
}
```

### AstChildren<N>

An iterator over child nodes of a particular AST type:

```rust
pub struct AstChildren<N> {
    inner: SyntaxNodeChildren,
    ph: PhantomData<N>,
}

impl<N: AstNode> Iterator for AstChildren<N> {
    type Item = N;
}
```

### Common Traits

| Trait | Methods | Description |
|-------|---------|-------------|
| `HasName` | `name() -> Option<ast::Name>` | Items with names |
| `HasVisibility` | `visibility() -> Option<ast::Visibility>` | Items with visibility |
| `HasLoopBody` | `loop_body()`, `label()` | Loop constructs |
| `HasArgList` | `arg_list() -> Option<ast::ArgList>` | Call expressions |
| `HasModuleItem` | `items() -> AstChildren<ast::Item>` | Containers with items |
| `HasGenericParams` | `generic_param_list()`, `where_clause()` | Generic items |
| `HasGenericArgs` | `generic_arg_list()` | Generic instantiations |
| `HasTypeBounds` | `type_bound_list()`, `colon_token()` | Type parameters |
| `HasAttrs` | `attrs()`, `has_atom_attr()` | Attributed items |
| `HasDocComments` | `doc_comments()` | Items with doc comments |

### Generated AST Types

The `ast::generated` module contains all Rust syntax node types. Key examples:

**Items:**
`Fn`, `Struct`, `Enum`, `Union`, `Trait`, `Impl`, `Module`, `Use`, `TypeAlias`, `Const`, `Static`, `MacroRules`, `MacroCall`, `ExternBlock`, `ExternCrate`

**Expressions:**
`Expr` (enum), `BinExpr`, `CallExpr`, `MethodCallExpr`, `IfExpr`, `MatchExpr`, `ForExpr`, `WhileExpr`, `LoopExpr`, `BlockExpr`, `PathExpr`, `RecordExpr`, `FieldExpr`, `IndexExpr`, `AwaitExpr`, `TryExpr`, `CastExpr`, `RefExpr`, `PrefixExpr`, `RangeExpr`, `Literal`, `ClosureExpr`, `ArrayExpr`, `TupleExpr`, `ParenExpr`, `LetExpr`, `UnderscoreExpr`

**Types:**
`Type` (enum), `PathType`, `TupleType`, `ArrayType`, `SliceType`, `RefType`, `PtrType`, `NeverType`, `FnPtrType`, `ForType`, `ImplTraitType`, `DynTraitType`, `InferType`, `MacroType`

**Patterns:**
`Pat` (enum), `IdentPat`, `TuplePat`, `RecordPat`, `SlicePat`, `PathPat`, `WildcardPat`, `RangePat`, `LiteralPat`, `OrPat`, `BoxPat`, `RefPat`, `MacroPat`, `RestPat`, `TupleStructPat`

---

## Syntax Pointers

Syntax trees are transient - they're created when needed and torn down to save memory. Pointers store *location* information that can be resolved later.

### SyntaxNodePtr

A lightweight pointer to a syntax node, storing kind and text range:

```rust
pub type SyntaxNodePtr = rowan::ast::SyntaxNodePtr<RustLanguage>;
```

### AstPtr<N>

A typed pointer that remembers the AST node type:

```rust
pub struct AstPtr<N: AstNode> {
    raw: SyntaxNodePtr,
    _ty: PhantomData<fn() -> N>,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(node: &N) -> AstPtr<N>` | Create pointer from node |
| `to_node` | `fn to_node(&self, root: &SyntaxNode) -> N` | Resolve to node |
| `syntax_node_ptr` | `fn syntax_node_ptr(&self) -> SyntaxNodePtr` | Get underlying pointer |
| `text_range` | `fn text_range(self) -> TextRange` | Get the text range |
| `cast` | `fn cast<U: AstNode>(self) -> Option<AstPtr<U>>` | Try to cast to another type |
| `kind` | `fn kind(&self) -> SyntaxKind` | Get the syntax kind |
| `upcast` | `fn upcast<M>(self) -> AstPtr<M>` | Upcast (where `N: Into<M>`) |
| `try_from_raw` | `fn try_from_raw(raw: SyntaxNodePtr) -> Option<AstPtr<N>>` | Create from raw pointer |

---

## SyntaxError

Represents parse errors, tokenization failures, or validation errors:

```rust
pub struct SyntaxError(String, TextRange);

impl SyntaxError {
    pub fn new(message: impl Into<String>, range: TextRange) -> Self;
    pub fn new_at_offset(message: impl Into<String>, offset: TextSize) -> Self;
    pub fn range(&self) -> TextRange;
    pub fn with_range(self, range: TextRange) -> Self;
}
```

Implements `Display`, `Error`.

---

## Algorithm Utilities (`algo` module)

Helpers for navigating and searching syntax trees.

| Function | Signature | Description |
|----------|-----------|-------------|
| `ancestors_at_offset` | `fn(node, offset) -> impl Iterator<Item = SyntaxNode>` | Ancestors at offset, sorted by length |
| `find_node_at_offset` | `fn<N: AstNode>(syntax, offset) -> Option<N>` | Find specific AST type at offset |
| `find_node_at_range` | `fn<N: AstNode>(syntax, range) -> Option<N>` | Find specific AST type covering range |
| `skip_trivia_token` | `fn(token, direction) -> Option<SyntaxToken>` | Skip to next non-trivia token |
| `skip_whitespace_token` | `fn(token, direction) -> Option<SyntaxToken>` | Skip whitespace only |
| `non_trivia_sibling` | `fn(element, direction) -> Option<SyntaxElement>` | Find non-trivia sibling |
| `least_common_ancestor` | `fn(u, v) -> Option<SyntaxNode>` | Find LCA of two nodes |
| `neighbor` | `fn<T: AstNode>(me, direction) -> Option<T>` | Find adjacent node of same type |
| `has_errors` | `fn(node) -> bool` | Check for ERROR children |
| `previous_non_trivia_token` | `fn(element) -> Option<SyntaxToken>` | Find previous non-trivia token |

---

## Tree Editing (`ted` module)

"Ed for trees" - primitive tree editing operations. Functions with `_raw` suffix insert as-is; unsuffixed versions fix up whitespace.

### Position

Represents where to insert:

```rust
impl Position {
    pub fn after(elem: impl Element) -> Position;
    pub fn before(elem: impl Element) -> Position;
    pub fn first_child_of(node: &impl Into<SyntaxNode>) -> Position;
    pub fn last_child_of(node: &impl Into<SyntaxNode>) -> Position;
    pub fn offset(&self) -> TextSize;
}
```

### Operations

| Function | Description |
|----------|-------------|
| `insert(position, elem)` | Insert element with whitespace fixup |
| `insert_raw(position, elem)` | Insert element as-is |
| `insert_all(position, elements)` | Insert multiple with fixup |
| `insert_all_raw(position, elements)` | Insert multiple as-is |
| `remove(elem)` | Remove element |
| `remove_all(range)` | Remove range of elements |
| `replace(old, new)` | Replace element |
| `replace_with_many(old, new)` | Replace with multiple |
| `replace_all(range, new)` | Replace range with multiple |
| `append_child(node, child)` | Append child with fixup |
| `append_child_raw(node, child)` | Append child as-is |
| `prepend_child(node, child)` | Prepend child with fixup |

---

## match_ast! Macro

Pattern match against AST types:

```rust
match_ast! {
    match node {
        ast::CallExpr(it) => { ... },
        ast::MethodCallExpr(it) => { ... },
        ast::MacroCall(it) => { ... },
        _ => None,
    }
}
```

---

## Public Submodules

| Module | Description |
|--------|-------------|
| `ast` | Abstract Syntax Tree types and traits |
| `ast::edit` | Functional-style AST editing helpers |
| `ast::edit_in_place` | In-place AST editing |
| `ast::make` | Factory functions for AST nodes |
| `ast::prec` | Operator precedence |
| `ast::syntax_factory` | Syntax factory for creating nodes |
| `algo` | Tree navigation algorithms |
| `ted` | Tree editing primitives |
| `syntax_editor` | Higher-level syntax editing |
| `hacks` | Workarounds for parser limitations |
| `utils` | Miscellaneous utilities |

---

## Usage in tarjanize

tarjanize-extract uses ra_ap_syntax for:

1. **Walking syntax trees** - Using `descendants()` and `preorder()` to find items
2. **AST types** - Using `ast::Fn`, `ast::Struct`, etc. for typed access
3. **Trait queries** - `HasName::name()` to get item names
4. **Text extraction** - `SyntaxNode::text()` to get source text
5. **Navigation** - `ancestors()`, `parent()` to find containing items

The syntax tree is obtained via `ra_ap_hir`'s `HasSource` trait, which provides access to source nodes for semantic items.
