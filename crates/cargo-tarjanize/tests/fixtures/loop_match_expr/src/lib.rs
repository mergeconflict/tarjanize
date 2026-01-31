//! Test: LoopMatch and ConstContinue expressions create edges.
//!
//! The `#[loop_match]` attribute is an experimental feature for state machines.
//! THIR represents the loop as `ExprKind::LoopMatch` and the state transitions
//! as `ExprKind::ConstContinue`.

#![feature(loop_match)]

pub enum State {
    Start,
    Middle,
    End,
}

pub fn helper_start() -> i32 {
    1
}

pub fn helper_middle() -> i32 {
    2
}

pub fn helper_end() -> i32 {
    3
}

pub fn state_machine() -> i32 {
    let mut result = 0;
    #[loop_match]
    loop {
        state = match state {
            State::Start => {
                result += helper_start();
                #[const_continue]
                State::Middle
            }
            State::Middle => {
                result += helper_middle();
                #[const_continue]
                State::End
            }
            State::End => {
                result += helper_end();
                break;
            }
        }
    }
    result
}
