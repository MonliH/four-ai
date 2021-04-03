use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

use super::N;
use crate::matrix;

#[derive(Clone, Deserialize, Serialize)]
pub enum Activation {
    Sigmoid,
    ELU,
    RELU,
}

impl Activation {
    pub fn from_string(s: &str) -> Activation {
        match s {
            "sigmoid" => Activation::Sigmoid,
            "elu" => Activation::ELU,
            "relu" => Activation::RELU,
            _ => panic!("invalid activation: {}", s),
        }
    }

    fn as_fn(&self) -> &(dyn Fn(N) -> N + Sync) {
        match self {
            Activation::Sigmoid => &&|x: N| 1.0 / (1.0 + std::f32::consts::E.powf(-x)),
            Activation::RELU => &&|x: N| if x > 0.0 { x } else { 0.0 },
            Activation::ELU => &&|x: N| {
                if x >= 0.0 {
                    x
                } else {
                    0.2 * (std::f32::consts::E.powf(x) - 1.0)
                }
            },
        }
    }
}

impl fmt::Debug for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Activation").finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NN {
    structure: Vec<usize>,
    activations: Vec<Activation>,
    pub weights: Vec<matrix::Matrix<N>>,
}

impl NN {
    pub fn new_rand(structure: Vec<usize>, activations: Vec<Activation>) -> Self {
        debug_assert_eq!(structure.len() - 1, activations.len());

        let mut weights: Vec<matrix::Matrix<N>> = Vec::with_capacity(structure.len());

        let mut rng = rand::thread_rng();
        for i in 0..structure.len() - 1 {
            weights.push(matrix::Matrix::from_rand(
                structure[i + 1],
                structure[i] + 1, // Add biases
                &mut || rng.gen_range(-1.0, 1.0),
            ));
        }

        NN {
            structure,
            weights,
            activations,
        }
    }

    pub fn forward(&self, input: Vec<N>) -> matrix::Matrix<N> {
        let mut activation = matrix::Matrix::into_row(input);

        for (weights, activation_fn) in self.weights.iter().zip(&self.activations) {
            activation.push(&mut vec![1.0]); // Push bias
            activation = weights * &activation;
            activation.map(&mut activation_fn.as_fn());
        }

        activation
    }
}
