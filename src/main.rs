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
use std::io::{stdin, stdout, Write};

fn main() {
    loop {
        print!("1) Play against the latest AI\n2) Play against another person (local)\n3) Train the ai\n\nEnter the code: ");
    
        stdout().flush().expect("Failed to flush to stdout");
        let mut command = String::new();
        stdin().read_line(&mut command).expect("Did not enter a correct string");

        command = command.chars().filter(|c| !c.is_whitespace()).collect();
        
        match &command[..] {
            "1" => {
                match game::play_against_ai(path::Path::new("./saves/gen")) {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("{}Failed: {}", RED!(), e);
                        std::process::exit(1);
                    }
                };
            }

            "2" => {
                game::start_two_player(); 
            }

            "3" => {
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
                    save_interval => 10,
                    file_path => path::PathBuf::from("./saves/gen")
                };

                let mut pool = ai::run::Pool::new(props);
                match pool.start() {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("{}Failed: {}", RED!(), e);
                        std::process::exit(1);
                    }
                }    

            }

            _ => {
                eprintln!("Invalid command: `{}`", command);
            }
        }
    }
}
