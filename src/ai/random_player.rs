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

    fn get_move(&self, _board: [[game::Spot; 6]; 7]) -> [N; 7] {
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    }
}
