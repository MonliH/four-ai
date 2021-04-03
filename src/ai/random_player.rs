use rand::Rng;

use super::{nn, Player, N};
use crate::game;

#[derive(Clone, Debug)]
pub struct RandomPlayer {}

impl RandomPlayer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Player for RandomPlayer {
    fn new_from_param(_structure: Vec<usize>, _activations: Vec<nn::Activation>) -> Self {
        Self {}
    }

    fn get_move(&self, _board: [[game::Spot; 6]; 7]) -> Vec<N> {
        let mut rng = rand::thread_rng();
        (0..42).map(|_| rng.gen_range(0.0, 1.0)).collect()
    }
}
