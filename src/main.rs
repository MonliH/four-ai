#[macro_use]
mod color;

#[macro_use]
mod ai;
mod game;
mod helpers;

mod matrix;

extern crate rand;
extern crate rayon;
extern crate serde;
extern crate serde_cbor;

use crate::ai::{
    pool::{Pool, PoolProperties},
    NNPlayer,
};

use std::fs::create_dir_all;
use std::io::{stdin, stdout, Write};
use std::path;

fn main() {
    loop {
        print!(
            r#"{}1) Play against the latest AI
{}2) Play against another person (local)
{}3) Train the ai
{}q) Exit{}

Enter the code: "#,
            BLUE!(),
            GREEN!(),
            RED!(),
            CYAN!(),
            RESET!()
        );

        stdout().flush().expect("Failed to flush to stdout");
        let mut command = String::new();
        stdin()
            .read_line(&mut command)
            .expect("Did not enter a correct string");

        command = command.chars().filter(|c| !c.is_whitespace()).collect();

        match &command[..] {
            "1" => {
                match game::play_against_ai::<NNPlayer>(path::Path::new("./saves/gen")) {
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
                    surviving_amount => 7,
                    mutation_amount => 3,
                    mutation_range => 0.001,
                    crossover_amount => 1,
                    structure => vec![42, 91, 91, 91, 7],
                    activations => vec! [
                        ai::nn::Activation::Sigmoid,
                        ai::nn::Activation::Sigmoid,
                        ai::nn::Activation::Sigmoid,
                        ai::nn::Activation::Sigmoid,
                    ],
                    generations => 100000000,
                    save_interval => 500,
                    compare_interval => 100000000000000,
                    file_path => path::PathBuf::from("./saves/gen")
                };

                create_dir_all("saves/").expect("Failed create new saves folder");

                let mut pool: Pool<NNPlayer> = Pool::new(props);
                match pool.start() {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("{}Failed: {}", RED!(), e);
                        std::process::exit(1);
                    }
                }
            }

            "q" | "Q" => {
                println!("Quitting...");
                std::process::exit(0);
            }

            _ => {
                println!("\x1b[2J\x1b[HInvalid option `{}`\n", command);
            }
        }
    }
}
