use crate::matrix;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Deserialize, Serialize)]
pub enum Activation {
    Sigmoid,
    ELU,
}

impl Activation {
    fn as_fn(&self) -> &(dyn Fn(f64) -> f64 + Sync) {
        match self {
            Activation::Sigmoid => &&|x: f64| 1.0 / (1.0 + std::f64::consts::E.powf(-x)),
            Activation::ELU => &&|x: f64| {
                if x >= 0.0 {
                    x
                } else {
                    0.2 * (std::f64::consts::E.powf(x) - 1.0)
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
    pub weights: Vec<matrix::Matrix<f64>>,
}

impl NN {
    pub fn new_rand(structure: Vec<usize>, activations: Vec<Activation>) -> Self {
        assert_eq!(structure.len() - 1, activations.len());

        let mut weights: Vec<matrix::Matrix<f64>> = Vec::with_capacity(structure.len());

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

    pub fn forward(&self, input: Vec<f64>) -> matrix::Matrix<f64> {
        let mut result = matrix::Matrix::into_row(input);

        for (layer, activation) in self.weights.iter().zip(&self.activations) {
            result.push(vec![1.0]); // Push bias
            result = layer * &result;
            result = result.mapped(&activation.as_fn());
        }

        result
    }
}
