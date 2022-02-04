use std::cmp::{Ordering, Reverse};
use std::error::Error;
use std::fs::{create_dir_all, File};
use std::path;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_cbor;

use super::{
    agent::{Agent, Player},
    nn, RandomPlayer, N,
};
use crate::game;
use crate::helpers;

#[derive(Serialize, Deserialize, Clone)]
pub struct PoolProperties {
    /// Amount of agents to retain per generations
    /// This means the number that die off is
    /// total_pos - surviving_amount
    pub surviving_amount: usize,

    /// Range of mutations on weights
    pub mutation_range: N,
    /// Probability that a mutation occurs
    pub mutation_prob: N,

    /// Number of crossed over agents
    pub crossover_size: usize,

    /// Total population of pool
    /// Most are killed off
    /// Calculated through (surviving_amount * surviving_amount - surviving_amount)* crossover_amount * mutation_amount
    pub population_size: usize,

    pub structure: Vec<usize>,
    pub activations: Vec<nn::Activation>,

    pub generations: isize,

    pub save_interval: isize,
    pub compare_interval: isize,
    pub file_path: path::PathBuf,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Pool<Plr: Player> {
    agents: Vec<Agent<Plr>>,
    generation: usize,
    properties: PoolProperties,
}

impl<'a, Plr> Pool<Plr>
where
    Plr: Player + Clone + Serialize + DeserializeOwned + Sync + Send,
{
    pub fn new(properties: PoolProperties) -> Pool<Plr> {
        let mut agents = Vec::with_capacity(properties.population_size);
        for _ in 0..properties.population_size {
            agents.push(Agent::new(Plr::new_from_param(
                properties.structure.clone(),
                properties.activations.clone(),
            )))
        }

        Pool {
            agents,
            generation: 0,
            properties,
        }
    }

    fn play<P1: Player, P2: Player>(
        &self,
        player1: &Agent<P1>,
        player2: &Agent<P2>,
    ) -> (game::Spot, usize) {
        let mut board = game::Board::new();
        let mut current_color = game::Spot::RED;
        let winner: game::Spot;

        'outer: loop {
            let mut temp = if current_color == game::Spot::RED {
                player1.player.get_move(board.positions)
            } else {
                player2.player.get_move(board.positions)
            };

            'inner: loop {
                let idx = temp
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
                    .unwrap();

                match board.insert_top(idx.0, current_color) {
                    (true, Some(win)) => {
                        winner = win;
                        break 'outer;
                    }
                    (true, None) => {
                        break 'inner;
                    }
                    (_, _) => {
                        temp[idx.0] = -100000.0;
                    }
                };
            }

            current_color = if current_color == game::Spot::RED {
                game::Spot::YELLOW
            } else {
                game::Spot::RED
            };
        }

        (winner, board.moves())
    }

    fn get_fitness<P1: Player, P2: Player>(
        &self,
        player1: &Agent<P1>,
        player2: &Agent<P2>,
    ) -> (i32, i32) {
        let win_amount = 1;
        let (winner1, moves1) = self.play(player1, player2);
        let (x, y) = match winner1 {
            game::Spot::RED => {
                // player1 wins
                (win_amount, -win_amount)
            }
            game::Spot::YELLOW => {
                // player2 wins
                (-win_amount, win_amount)
            }
            game::Spot::EMPTY => {
                // tie
                (0, 0)
            }
        };

        let (winner2, moves2) = self.play(player2, player1);
        let (temp2, temp1) = match winner2 {
            game::Spot::RED => {
                // player1 wins
                (win_amount, -win_amount)
            }
            game::Spot::YELLOW => {
                // player2 wins
                (-win_amount, win_amount)
            }
            game::Spot::EMPTY => {
                // tie
                (0, 0)
            }
        };

        let move_fitness = 0;
        (x + temp1 + move_fitness, y + temp2 + move_fitness)
    }

    fn mutate_crossover(&mut self, new_pop: &mut Vec<Agent<Plr>>) {
        'crossover: for i in 0..new_pop.len() {
            for k in 0..new_pop.len() {
                if i != k {
                    if self.agents.len() < self.properties.crossover_size {
                        let mut new_agent = new_pop[i].clone();
                        new_agent.player.crossover(&new_pop[k].player);
                        self.agents.push(new_agent);
                    } else {
                        break 'crossover;
                    }
                }
            }
        }

        'copy: loop {
            for net in new_pop.iter() {
                if !(self.agents.len() >= self.properties.population_size) {
                    break 'copy;
                }
                self.agents.push(net.clone());
            }
        }

        for agent in self.agents.iter_mut() {
            agent.player.mutate(
                self.properties.mutation_range,
                self.properties.mutation_prob,
            );
            agent.fitness = 0;
        }
    }

    #[inline(always)]
    pub fn get_range(s: usize, e: isize) -> Box<dyn Iterator<Item = usize>> {
        if e <= -1 {
            Box::new((s..).into_iter())
        } else {
            Box::new((s..(e as usize)).into_iter())
        }
    }

    #[inline(always)]
    pub fn training_loop(&mut self, start: usize) -> Result<(), Box<dyn Error>> {
        for gen in Self::get_range(start, self.properties.generations) {
            self.generation = gen;

            // Generation loop
            let fitness_diffs = Arc::new(Mutex::new(vec![0; self.agents.len()]));
            (0..self.agents.len()).into_par_iter().for_each(|i| {
                let mut i_fitness_delta = 0;
                for j in 0..self.agents.len() {
                    if i != j {
                        // Play against each other
                        let fitnesses = self.get_fitness(&self.agents[i], &self.agents[j]);
                        i_fitness_delta += fitnesses.0;
                        let mut obj = fitness_diffs.lock().unwrap();
                        obj[j] += fitnesses.1;
                        std::mem::drop(obj);
                    }
                }

                let mut obj = fitness_diffs.lock().unwrap();
                obj[i] += i_fitness_delta;
                std::mem::drop(obj);
            });

            for (i, fitness_dif) in fitness_diffs.lock().unwrap().iter().enumerate() {
                self.agents[i].fitness += fitness_dif;
            }

            self.agents.sort_unstable_by_key(|x| Reverse(x.fitness));
            let mut new_pop = self
                .agents
                .drain(0..self.properties.surviving_amount)
                .collect::<Vec<_>>();
            self.agents.clear();

            if self.properties.save_interval >= 0
                && self.generation != 0
                && self.generation % (self.properties.save_interval as usize) == 0
            {
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

            if self.properties.compare_interval >= 0
                && self.generation != 0
                && self.generation % (self.properties.compare_interval as usize) == 0
            {
                print!(
                    "{}Calculating fitness relative to dumb agent...{} ",
                    BLUE!(),
                    RESET!()
                );
                let mut random_fitness = 0;
                for agent in new_pop[0..1].iter() {
                    random_fitness += self.get_fitness(agent, &Agent::new(RandomPlayer::new())).0;
                }
                println!(
                    "{}Top population has a fitness of {} against dumb agent.{}",
                    GREEN!(),
                    random_fitness,
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
                let mut new_pop: Vec<Agent<Plr>> = serde_cbor::from_reader(file)?;
                self.agents.clear();
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
                    self.properties.population_size,
                    RESET!()
                );
                0
            };

        println!("");

        self.training_loop(start)
    }
}
