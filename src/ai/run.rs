use crate::ai::nn;
use crate::game;
use crate::helpers;
use crate::rand::Rng;

use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use serde_cbor;

use std::cmp::Ordering;
use std::error::Error;
use std::fs::{create_dir_all, File};
use std::path;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Agent {
    fitness: i32,
    nn: nn::NN,
}

impl Agent {
    fn new(structure: Vec<usize>, activations: Vec<nn::Activation>) -> Agent {
        Agent {
            fitness: 0,
            nn: nn::NN::new_rand(structure, activations),
        }
    }

    pub fn get_move(&self, board: [[game::Spot; 6]; 7]) -> Vec<f64> {
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

    fn crossover(&mut self, other: &Agent) {
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

#[derive(Serialize, Deserialize)]
pub struct PoolProperties {
    /// Amount of agents to retain per generations
    /// This means the number that die off is
    /// total_pos - surviving_amount
    pub surviving_amount: usize,

    /// Amount of mutations to do per agent
    pub mutation_amount: usize,

    /// Range of mutations on weights
    pub mutation_range: f64,

    /// Amount of crossovers to do per agent
    pub crossover_amount: usize,

    /// Total population of pool
    /// Most are killed off
    /// Calculated through (surviving_amount * surviving_amount - surviving_amount)* crossover_amount * mutation_amount
    pub total_pop: Option<usize>,

    pub structure: Vec<usize>,
    pub activations: Vec<nn::Activation>,

    pub generations: usize,

    pub save_interval: usize,
    pub file_path: path::PathBuf,
}

#[macro_export]
macro_rules! pool_props {
    ($($prop_name:ident => $value:expr),+) => {
        {
            $(let $prop_name = $value;)+
            let mut properties = PoolProperties {
                $($prop_name,)+
                total_pop: None
            };

            properties.total_pop = Some((properties.surviving_amount * properties.surviving_amount - properties.surviving_amount) * properties.mutation_amount * properties.crossover_amount + properties.surviving_amount);
            properties
        }
    };
}

#[derive(Serialize, Deserialize)]
pub struct Pool {
    agents: Vec<Agent>,
    generation: usize,
    properties: PoolProperties,
}

impl Pool {
    pub fn new(properties: PoolProperties) -> Self {
        let mut agents = Vec::with_capacity(properties.total_pop.unwrap());
        for _ in 0..properties.total_pop.unwrap() {
            agents.push(Agent::new(
                properties.structure.clone(),
                properties.activations.clone(),
            ))
        }

        Pool {
            agents,
            generation: 0,
            properties,
        }
    }

    fn play(&self, player1: &Agent, player2: &Agent) -> (game::Spot, i32) {
        let mut board = game::Board::new();
        let mut current_color = game::Spot::RED;
        let mut moves = 0;
        let winner: game::Spot;

        'outer: loop {
            let temp = if current_color == game::Spot::RED {
                player1.get_move(board.positions)
            } else {
                player2.get_move(board.positions)
            };

            let mut ai_moves = temp.iter().enumerate().collect::<Vec<_>>();
            moves += 1;
            'inner: loop {
                let idx = ai_moves
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                    .unwrap_or((0, &(0, &1.0)));

                match board.insert_top((idx.1).0, current_color) {
                    (true, Some(win)) => {
                        winner = win;
                        break 'outer;
                    }
                    (true, None) => {
                        break 'inner;
                    }
                    (_, _) => {
                        let idx = idx.0;
                        ai_moves.remove(idx);
                    }
                };
            }

            current_color = if current_color == game::Spot::RED {
                game::Spot::YELLOW
            } else {
                game::Spot::RED
            };
        }

        (winner, moves)
    }

    fn get_fitness(&self, i: usize, j: usize) -> (i32, i32) {
        let player1 = &self.agents[i];
        let player2 = &self.agents[j];

        let (x, y) = match self.play(player1, player2) {
            (game::Spot::RED, moves) => {
                // player1 wins
                (45 - moves, 0)
            }
            (game::Spot::YELLOW, moves) => {
                // player2 wins
                (0, 45 - moves)
            }
            (game::Spot::EMPTY, moves) => {
                // tie
                (45 - moves, 45 - moves)
            }
        };

        let (temp2, temp1) = match self.play(player2, player1) {
            (game::Spot::RED, moves) => {
                // player1 wins
                (45 - moves, 0)
            }
            (game::Spot::YELLOW, moves) => {
                // player2 wins
                (0, 45 - moves)
            }
            (game::Spot::EMPTY, moves) => {
                // tie
                (45 - moves, 45 - moves)
            }
        };

        (x + temp1, y + temp2)
    }

    fn mutate_crossover(&mut self, new_pop: &mut Vec<Agent>) {
        for i in 0..new_pop.len() {
            for k in 0..new_pop.len() {
                if i != k {
                    for _ in 0..self.properties.crossover_amount {
                        let mut breed_agent = new_pop[i].clone();
                        breed_agent.crossover(&new_pop[k]);

                        for _ in 0..self.properties.mutation_amount {
                            // Yes 4 nested for loops
                            let mut mutated_agent = breed_agent.clone();
                            mutated_agent.mutate(self.properties.mutation_range);

                            mutated_agent.fitness = 0;
                            self.agents.push(mutated_agent);
                        }
                    }
                }
            }
        }

        self.agents.append(new_pop);
    }

    pub fn start(&mut self) -> Result<(), Box<dyn Error>> {
        println!("{}Looking for previous saves...{}", BLUE!(), RESET!());
        let start: usize =
            if let Some(val) = helpers::get_max_generation(&self.properties.file_path)? {
                let filename = val.file_name();
                let os_to_str = filename.to_str().unwrap();
                let gen = os_to_str
                    .split("_")
                    .last()
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                print!(
                    "{}Detected generation {}, starting from there... {}",
                    BLUE!(),
                    gen,
                    RESET!()
                );
                let file = File::open(val.path())?;
                let mut new_pop = serde_cbor::from_reader(file)?;
                self.mutate_crossover(&mut new_pop);
                println!("{}Loaded generations{}", BLUE!(), RESET!());
                println!(
                    "{}Starting with a population of {}{}",
                    GREEN!(),
                    self.agents.len(),
                    RESET!()
                );
                gen
            } else {
                println!(
                    "{}Starting with a population of {}{}",
                    GREEN!(),
                    self.properties.total_pop.unwrap(),
                    RESET!()
                );
                0
            };

        println!("");

        for gen in start..self.properties.generations {
            self.generation = gen;

            // Generation loop
            let fitness_diffs = Arc::new(Mutex::new(vec![0; self.agents.len()]));
            (0..self.agents.len()).into_par_iter().for_each(|i| {
                {
                    for j in 0..self.agents.len() {
                        if i != j {
                            // Play against each other
                            let fitnesses = self.get_fitness(i, j);
                            let mut obj = fitness_diffs.lock().unwrap();
                            obj[i] += fitnesses.0;
                            obj[j] += fitnesses.1;
                        }
                    }
                }
            });

            for (i, fitness_dif) in fitness_diffs.lock().unwrap().iter().enumerate() {
                self.agents[i].fitness += fitness_dif;
            }

            self.agents.sort_unstable_by_key(|x| x.fitness);
            let mut new_pop = self
                .agents
                .drain(0..self.properties.surviving_amount)
                .collect::<Vec<_>>();
            self.agents.clear();

            if self.generation % self.properties.save_interval == 0 {
                print!(
                    "{}Writing generation {}... {}",
                    BLUE!(),
                    self.generation,
                    RESET!()
                );
                create_dir_all(
                    self.properties
                        .file_path
                        .parent()
                        .unwrap_or(path::Path::new("")),
                )?;
                let path = format!(
                    "{}_{}",
                    self.properties.file_path.to_str().unwrap(),
                    self.generation
                );
                let file = File::create(&path[..])?;

                serde_cbor::to_writer(file, &new_pop)?;
                println!(
                    "{}Done writing generation {}{}",
                    BLUE!(),
                    self.generation,
                    RESET!()
                );
            }

            print!(
                "{}Top fitness: {}. {}",
                GREEN!(),
                new_pop.first().unwrap().fitness,
                RESET!()
            );
            self.mutate_crossover(&mut new_pop);

            println!(
                "{}Generation {} done.{}",
                CYAN!(),
                self.generation,
                RESET!()
            );
        }
        Ok(())
    }
}
