pub mod agent;
mod nn_player;
mod random_player;

pub mod nn;

#[macro_use]
pub mod pool;

use agent::Player;
pub use nn_player::NNPlayer;
pub use random_player::RandomPlayer;
