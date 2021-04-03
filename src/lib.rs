// We can disable warnings for lib.rs, but still recive them in main.rs
#![allow(warnings)]
#[macro_use]
mod color;

#[macro_use]
pub mod ai;

mod game;
mod helpers;

pub mod matrix;
