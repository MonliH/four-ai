#[macro_use]
mod color;

#[macro_use]
pub mod ai;
mod game;
pub mod helpers;

extern crate rand;
extern crate rayon;
extern crate serde;
extern crate serde_cbor;

use crate::ai::run::PoolProperties;
use std::path;

fn main() {
    let props = pool_props! {
        surviving_amount => 5,
        mutation_amount => 3,
        mutation_range => 0.075,
        crossover_amount => 1,
        structure => vec![42, 128, 256, 128, 7],
        activations => vec! [
            ai::nn::Activation::Sigmoid,
            ai::nn::Activation::ELU,
            ai::nn::Activation::ELU,
            ai::nn::Activation::Sigmoid,
        ],
        generations => 10000,
        save_amount => 10,
        file_path => path::PathBuf::from("./saves/gen")
    };

    let mut pool = ai::run::Pool::new(props);
    match pool.start() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("{}Failed: {}", RED!(), e);
            std::process::exit(1);
        }
    };

    /*match game::play_against_ai(path::Path::new("./saves/gen")) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("{}Failed: {}", RED!(), e);
            std::process::exit(1);
        }
    };*/
}
