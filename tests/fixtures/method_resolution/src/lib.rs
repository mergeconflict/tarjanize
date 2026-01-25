trait Foo {
    fn default_method(&self) {}
    fn required(&self);
}

struct Bar;
impl Foo for Bar {
    fn required(&self) {}
    // default_method not overridden
}

struct Baz;
impl Foo for Baz {
    fn default_method(&self) {}
    fn required(&self) {}
}

fn caller() {
    let bar = Bar;
    bar.default_method();  // Case 1: uses trait default
    bar.required();        // Case 2: uses impl (required)

    let baz = Baz;
    baz.default_method();  // Case 3: uses impl override
    baz.required();        // Case 4: uses impl (required)
}
