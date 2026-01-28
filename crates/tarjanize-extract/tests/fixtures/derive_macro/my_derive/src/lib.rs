use proc_macro::TokenStream;

/// A simple derive macro that does nothing (identity).
///
/// Used to test whether derive macros are captured as dependencies.
#[proc_macro_derive(MyDerive)]
pub fn my_derive(_input: TokenStream) -> TokenStream {
    TokenStream::new()
}
