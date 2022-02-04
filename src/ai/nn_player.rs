use rand::Rng;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;

use super::{nn, Player, N};
use crate::game;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NNPlayer {
    nn: nn::NN,
}

impl Player for NNPlayer {
    fn new_from_param(structure: Vec<usize>, activations: Vec<nn::Activation>) -> Self {
        Self {
            nn: nn::NN::new_rand(structure, activations),
        }
    }

    fn get_move(&self, board: [[game::Spot; 6]; 7]) -> [N; 7] {
        let flattened_board = board
            .iter()
            .flatten()
            .map(|x| x.into_rep())
            .collect::<Vec<_>>();

        self.nn
            .forward(flattened_board)
            .T()
            .values
            .try_into()
            .unwrap()
    }

    fn mutate(&mut self, mutation_range: N, mutation_prob: N) {
        let mut rng = rand::thread_rng(); //rng::thread_rng();
        for i in 0..self.nn.weights.len() {
            self.nn.weights[i].map(&mut |x| {
                if rng.gen::<N>() < mutation_prob {
                    x + rng.gen_range(-mutation_range, mutation_range)
                } else {
                    x
                }
            });
        }
    }

    fn crossover(&mut self, other: &Self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.nn.weights.len() {
            if rng.gen::<f32>() < 0.5 {
                self.nn.weights[i] = other.nn.weights[i].clone();
            }
        }
    }
}
