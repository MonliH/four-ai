use crate::ai::nn;
use crate::game;

use serde::{Deserialize, Serialize};

pub trait Player {
    fn new_from_param(structure: Vec<usize>, activations: Vec<nn::Activation>) -> Self;
    fn mutate(&mut self, _mutation_range: f64) {}
    fn crossover(&mut self, _other: &Self) {}
    fn get_move(&self, board: [[game::Spot; 6]; 7]) -> Vec<f64>;
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Agent<Plr: Player> {
    pub player: Plr,
    pub fitness: i32,
}

impl<Plr> Agent<Plr>
where
    Plr: Player,
{
    pub fn new(player: Plr) -> Self {
        Self { fitness: 0, player }
    }
}

