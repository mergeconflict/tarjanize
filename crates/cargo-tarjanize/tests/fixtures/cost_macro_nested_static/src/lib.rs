//! Test: Macro-generated nested statics aggregate to parent.

macro_rules! with_counter {
    ($name:ident, $body:expr) => {
        pub fn $name() -> i32 {
            static COUNTER: std::sync::atomic::AtomicI32 =
                std::sync::atomic::AtomicI32::new(0);
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            $body
        }
    };
}

with_counter!(with_macro_static, 42);
