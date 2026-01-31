//! Test: Labels are not dependency targets.

pub struct S;

pub fn caller() -> S {
    'outer: loop {
        break 'outer S;
    }
}
