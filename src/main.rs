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

use ai::nn::Activation;
use clap::Parser;
use std::{fs::create_dir_all, path::PathBuf};

const VERSION: &'static str = env!("CARGO_PKG_VERSION");
const AUTHOR: &'static str = env!("CARGO_PKG_AUTHORS");

#[derive(Parser, Debug)]
#[clap(
    version = VERSION,
    author = AUTHOR,
)]
/// Neural networks trained with genetic algorithm to play connect four
struct Opts {
    #[clap(subcommand)]
    subcmd: Subcommands,
}

#[derive(Parser, Debug)]
enum Subcommands {
    #[clap(about = "Train the neural network")]
    Train(Train),
    #[clap(about = "Play against the neural network")]
    PlayAi(PlayAi),
    #[clap(about = "Play against another play, locallaly (no ai)")]
    PlayLocal(PlayLocal),
}

#[derive(Parser, Debug)]
struct PlayLocal {}

#[derive(Parser, Debug)]
struct PlayAi {
    #[clap(short = 'n', long = "generation", default_value = "-1")]
    /// Generation to play against, `-1` for the lastest generation
    generation_num: i32,

    #[clap(short = 'f', long = "ai-first")]
    /// Make the AI go first (i.e. play as yellow)
    ai_first: bool,

    #[clap(short = 'p', long = "save-path", default_value = "./saves/gen")]
    /// Generation path to load from. Generation number is added to the end of the filename.
    /// E.g. `./saves/gen2500` is loaded for generation 2500 if `save-path` is `./saves/gen`
    save_path: PathBuf,
}

#[derive(Parser, Debug)]
struct Train {
    #[clap(short = 'p', long = "save-path", default_value = "./saves/gen")]
    /// Generation save path.
    ///
    /// Generation number is added to the end of the filename.
    /// E.g. `./saves/gen2500` is saved for generation 2500 if `save-path` is `./saves/gen`
    save_path: PathBuf,

    #[clap(short = 's', long = "surviving", default_value = "5")]
    /// The surviving population that lives into the next generation
    surviving: usize,
    #[clap(short = 'M', long = "mutation-range", default_value = "0.015")]
    /// Mutation range, i.e. how much to mutate each weight by
    mutation_range: f32,
    #[clap(short = 'P', long = "mutation-prob", default_value = "0.05")]
    /// Probablity of mutation, i.e. how often to mutate each weight
    mutation_prob: f32,
    #[clap(short = 'c', long = "crossover-size", default_value = "30")]
    /// Number of agents that result from crossover
    crossover_size: usize,
    #[clap(short = 'p', long = "population-size", default_value = "200")]
    /// Total population size
    population_size: usize,
    #[clap(short = 'g', long = "generations", default_value = "-1")]
    /// Number of generations to train for.
    /// Use `-1` to train indefinitely, until stopped (i.e. interrupt)
    generations: isize,
    #[clap(short = 'i', long = "save-interval", default_value = "250")]
    /// Interval to save the generations.
    /// Use `-1` to never save.
    save_interval: isize,
    #[clap(short = 'I', long = "compare-interval", default_value = "100")]
    /// Interval to compare the neural network population to a random agent.
    /// Use `-1` to never compare.
    compare_interval: isize,
    #[clap(short = 'S', long = "structure", multiple_values=true, default_values = &["42", "128", "256", "128", "7"])]
    /// Structure of the neural network. Must begin with 42 and end with 7 (board input and
    /// outputs)
    structure: Vec<usize>,
    #[clap(
        short = 'a',
        long = "activations",
        multiple_values=true,
        default_values = &["sigmoid", "sigmoid", "sigmoid", "sigmoid"],
        possible_values = &["sigmoid", "elu", "relu"]
    )]
    /// Activation functions to use between layers.
    /// Must be the same length as the structure minus 1.
    activations: Vec<String>,
}

fn main() {
    let opt = Opts::parse();
    match opt.subcmd {
        Subcommands::Train(config) => {
            create_dir_all(
                config
                    .save_path
                    .parent()
                    .expect("Invalid save path provided"),
            )
            .expect("Failed create new saves folder");

            let activations = config
                .activations
                .into_iter()
                .map(|a_str| Activation::from_string(&a_str))
                .collect::<Vec<_>>();

            let props = PoolProperties {
                population_size: config.population_size,
                mutation_prob: config.mutation_prob,
                surviving_amount: config.surviving,
                mutation_range: config.mutation_range,
                crossover_size: config.crossover_size,
                structure: config.structure,
                activations: activations,
                generations: config.generations,
                save_interval: config.save_interval,
                compare_interval: config.compare_interval,
                file_path: config.save_path,
            };

            let mut pool: Pool<NNPlayer> = Pool::new(props);
            match pool.start() {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("{}Failed: {}", RED!(), e);
                    std::process::exit(1);
                }
            }
        }
        Subcommands::PlayAi(config) => {
            match game::play_against_ai::<NNPlayer>(&config.save_path, config.ai_first) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("{}Failed: {}", RED!(), e);
                    std::process::exit(1);
                }
            };
        }
        Subcommands::PlayLocal(_) => {
            game::start_two_player();
        }
    }
}
