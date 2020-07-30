use crate::game;
use rand::Rng;

use crate::ai::nn;
use crate::ai::Player;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NNAgent {
    fitness: i32,
    nn: nn::NN,
}

impl Player for NNAgent {
    fn new_from_param(structure: Vec<usize>, activations: Vec<nn::Activation>) -> Self {
        NNAgent {
            fitness: 0,
            nn: nn::NN::new_rand(structure, activations),
        }
    }

    fn get_move(&self, board: [[game::Spot; 6]; 7]) -> Vec<f64> {
        let flattened_board = board
            .iter()
            .flatten()
            .map(|x| x.into_rep())
            .collect::<Vec<_>>();

        self.nn.forward(flattened_board).T().values.remove(0)
    }

    fn mutate(&mut self, mutation_range: f64) {
        let mut rng = rand::thread_rng();
        for i in 0..self.nn.weights.len() {
            self.nn.weights[i].map(&mut |x| x + rng.gen_range(-mutation_range, mutation_range));
        }
    }

    fn crossover(&mut self, other: &Self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.nn.weights.len() {
            self.nn.weights[i].map_enumerate(&mut |j, k, x| {
                if rng.gen_range(0.0, 1.0) > 0.7 {
                    other.nn.weights[i].get(j, k)
                } else {
                    x
                }
            });
        }
    }
}
