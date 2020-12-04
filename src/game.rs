use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead};
use std::path;
use std::process;

use crate::ai::agent::Player;
use crate::ai::nn_agent::NNAgent;
use crate::helpers;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Spot {
    EMPTY,
    RED,
    YELLOW,
}

impl fmt::Display for Spot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}{}",
            match self {
                Spot::EMPTY => "",
                Spot::RED => RED!(),
                Spot::YELLOW => YELLOW!(),
            },
            match self {
                Spot::EMPTY => "  ",
                Spot::RED => "██",
                Spot::YELLOW => "██",
            },
            RESET!()
        )
    }
}

impl Spot {
    fn display(&self) -> &'static str {
        match self {
            Spot::RED => concat!(BOLD!(), RED!(), "RED", RESET!()),
            Spot::YELLOW => concat!(BOLD!(), YELLOW!(), "YELLOW", RESET!()),
            Spot::EMPTY => "",
        }
    }

    pub fn into_rep(&self) -> f64 {
        match self {
            Spot::RED => 1.0,
            Spot::YELLOW => -1.0,
            Spot::EMPTY => 0.0,
        }
    }
}

pub struct Board {
    pub positions: [[Spot; 6]; 7],
    highest_pieces: [isize; 7],
    dimensions: (usize, usize),
    moves: usize,
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows: [String; 6] = Default::default();
        writeln!(
            f,
            " {} ",
            (0..self.positions.len() * 5 - 1)
                .map(|x| if x % 5 == 0 {
                    ((x + 1) / 5 + 1).to_string()
                } else {
                    " ".to_string()
                })
                .collect::<String>()
        )?;
        writeln!(
            f,
            "┏{}┓",
            (0..self.positions.len() * 5 - 1)
                .map(|x| if (x + 1) % 5 == 0 { "┳" } else { "━" })
                .collect::<String>()
        )?;

        for col in &self.positions {
            for (i, value) in col.iter().enumerate() {
                rows[i] += &value.to_string()[..];
                rows[i] += " ┃ ";
            }
        }
        for row in &rows {
            writeln!(f, "┃ {}", row)?;
        }

        writeln!(
            f,
            "┗{}┛",
            (0..self.positions.len() * 5 - 1)
                .map(|x| if (x + 1) % 5 == 0 { "┻" } else { "━" })
                .collect::<String>()
        )?;

        Ok(())
    }
}

impl Board {
    pub fn new() -> Self {
        let rows = [Spot::EMPTY; 6];
        let positions = [rows; 7];
        let highest_pieces = [5; 7];
        let dimensions: (usize, usize) = (6, 7);

        Board {
            positions,
            highest_pieces,
            dimensions,
            moves: 0,
        }
    }

    fn change_position(&mut self, x: usize, y: usize, spot: Spot) {
        self.positions[x][y] = spot;
    }

    fn check_four_consecutive(&self, pieces: Vec<Spot>) -> Option<Spot> {
        match pieces
            .windows(4)
            .map(|arr| {
                if arr.windows(2).all(|val| val[0] == val[1]) {
                    // All values are the same, win
                    Some(arr[0])
                } else {
                    None
                }
            })
            .filter_map(|x| x)
            .collect::<Vec<_>>()[..]
        {
            [winner] if winner != Spot::EMPTY => Some(winner),
            _ => None,
        }
    }

    fn check_win(&self, column: usize, row: usize) -> Option<Spot> {
        // Horizontal Check
        match self.check_four_consecutive(
            (0..self.dimensions.1)
                .map(|column_no| self.positions[column_no][row])
                .collect::<Vec<_>>(),
        ) {
            Some(winner) => {
                return Some(winner);
            }
            _ => {}
        };

        // Vertical Check
        match self.check_four_consecutive(
            (0..self.dimensions.0)
                .map(|row_no| self.positions[column][row_no])
                .collect::<Vec<_>>(),
        ) {
            Some(winner) => {
                return Some(winner);
            }
            _ => {}
        };

        // Forward slash diagonal /
        let mut col_calc: isize = column as isize - (self.dimensions.0 as isize - 1 - row as isize);
        let mut row_calc: isize = row as isize;

        if col_calc < 0 {
            row_calc += col_calc + 2;
            col_calc = 0;
        }

        let calculated_pos: (usize, isize) = (
            col_calc as usize,
            (row_calc - (col_calc as isize - column as isize)),
        );

        match self.check_four_consecutive(
            (0..(calculated_pos.1 + 1))
                .rev()
                .map(|row_no| {
                    self.positions
                        .get(calculated_pos.0 + row_no as usize)?
                        .get(calculated_pos.1 as usize - row_no as usize)
                        .copied()
                })
                .filter_map(|x| x)
                .collect::<Vec<_>>(),
        ) {
            Some(winner) => {
                return Some(winner);
            }
            _ => {}
        };

        // Back slash diagonal \
        let mut col_calc: usize = (self.dimensions.0 - 1 - row) + column;
        let mut row_calc: usize = row;

        if col_calc > self.dimensions.1 - 1 {
            row_calc += col_calc - self.dimensions.1 + 1;
            col_calc = self.dimensions.1 - 1;
        }

        let calculated_pos: (usize, usize) = (col_calc, self.dimensions.1 - 2 - (row_calc - row));
        match self.check_four_consecutive(
            (0..(calculated_pos.1 + 1))
                .map(|row_no| {
                    if row_no <= calculated_pos.0 && row_no <= calculated_pos.1 {
                        self.positions
                            .get(calculated_pos.0 - row_no)?
                            .get(calculated_pos.1 - row_no)
                            .copied()
                    } else {
                        None
                    }
                })
                .filter_map(|x| x)
                .collect::<Vec<_>>(),
        ) {
            Some(winner) => {
                return Some(winner);
            }
            _ => {}
        };

        None
    }

    pub fn insert_top(&mut self, column: usize, spot: Spot) -> (bool, Option<Spot>) {
        let highest = self.highest_pieces[column];
        if highest != -1 {
            self.change_position(column, highest as usize, spot);
            self.highest_pieces[column] -= 1;
            self.moves += 1;
            (true, self.check_win(column, highest as usize))
        } else if self.moves >= self.dimensions.0 * self.dimensions.1 {
            (true, Some(Spot::EMPTY))
        } else {
            (false, None)
        }
    }
}

pub fn start_two_player() {
    let mut board = Board::new();
    let mut current_player = Spot::RED;
    let mut fail = "";

    loop {
        println!(
            "\x1b[2J\x1b[H{}{}It's {}'s turn!",
            board,
            fail,
            current_player.display()
        );
        eprint!("Enter your move (between 1-7): ");
        let mut column = String::new();
        let stdin = io::stdin();
        stdin.lock().read_line(&mut column).unwrap();
        if column.ends_with('\n') {
            column.pop();
            if column.ends_with('\r') {
                column.pop();
            }
        }
        match column.parse::<usize>() {
            Ok(val) if val >= 1 && val <= 7 => {
                fail = "";
                match board.insert_top(val - 1, current_player) {
                    (false, _) => {
                        fail = concat!(BOLD!(), "That column in full. Try again! ", RESET!());
                        continue;
                    }
                    (true, Some(_)) => {
                        // Winner
                        break;
                    }
                    (true, None) => {
                        // Continue playing
                    }
                };
            }
            _ => {
                fail = concat!(
                    BOLD!(),
                    "Invalid input! Please enter an number between 1-7. ",
                    RESET!()
                );
                continue;
            }
        }

        current_player = if current_player == Spot::RED {
            Spot::YELLOW
        } else {
            Spot::RED
        };
    }

    println!("\x1b[2J\x1b[H{}{} Wins!", board, current_player.display());
}

pub fn play_against_ai(ai_path: &path::Path) -> Result<(), Box<dyn Error>> {
    let mut board = Board::new();
    let mut current_player = Spot::RED;
    let ai_turn = Spot::YELLOW;
    let mut fail = "";

    let nn: NNAgent = match helpers::get_max_generation(ai_path)? {
        Some(dir) => {
            let path = dir.path();
            let file = File::open(path)?;
            serde_cbor::from_reader::<Vec<NNAgent>, _>(file)?.remove(0)
        }
        None => {
            println!("Error, no file exists.");
            process::exit(1);
        }
    };

    'outer: loop {
        println!(
            "\x1b[2J\x1b[H{}{}It's {}'s turn!",
            board,
            fail,
            current_player.display()
        );
        eprint!("Enter your move (between 1-7): ");

        if current_player != ai_turn {
            let mut column = String::new();
            let stdin = io::stdin();
            stdin.lock().read_line(&mut column).unwrap();
            if column.ends_with('\n') {
                column.pop();
                if column.ends_with('\r') {
                    column.pop();
                }
            }

            match board.insert_top(
                match column.parse::<usize>() {
                    Ok(val) if val >= 1 && val <= 7 => {
                        fail = "";
                        val - 1
                    }
                    _ => {
                        fail = concat!(
                            BOLD!(),
                            "Invalid input! Please enter an number between 1-7. ",
                            RESET!()
                        );
                        continue;
                    }
                },
                current_player,
            ) {
                (false, _) => {
                    fail = concat!(BOLD!(), "That column is full. Try again! ", RESET!());
                    continue;
                }
                (true, Some(_)) => {
                    // Winner
                    break 'outer;
                }
                (true, None) => {
                    // Continue playing
                }
            };
        } else {
            let moves = nn.get_move(board.positions);
            let mut nn_moves = moves.iter().enumerate().collect::<Vec<_>>();
            'inner: loop {
                let idx = nn_moves
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                    .unwrap_or((0, &(0, &1.0)));

                match board.insert_top((idx.1).0, current_player) {
                    (true, Some(_)) => {
                        break 'outer;
                    }
                    (true, None) => {
                        break 'inner;
                    }
                    (_, _) => {
                        let idx = idx.0;
                        nn_moves.remove(idx);
                    }
                };
            }
        }

        current_player = if current_player == Spot::RED {
            Spot::YELLOW
        } else {
            Spot::RED
        };
    }

    println!("\x1b[2J\x1b[H{}{} Wins!", board, current_player.display());

    Ok(())
}

#[cfg(test)]
mod game_tests {
    use super::*;

    #[test]
    fn forward_diagonal_1() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, Some(Spot::RED)), board.insert_top(4, Spot::RED));
    }

    #[test]
    fn forward_diagonal_2() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(1, Spot::RED));
        assert_eq!((true, None), board.insert_top(1, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(1, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(3, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::YELLOW));

        assert_eq!((true, None), board.insert_top(0, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(1, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(2, Spot::YELLOW));
        assert_eq!(
            (true, Some(Spot::YELLOW)),
            board.insert_top(3, Spot::YELLOW)
        );
    }

    #[test]
    fn forward_diagonal_3() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(1, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(1, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(2, Spot::YELLOW));
        assert_eq!(
            (true, Some(Spot::YELLOW)),
            board.insert_top(3, Spot::YELLOW)
        );
    }

    #[test]
    fn backward_diagonal_1() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(5, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(6, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(5, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(4, Spot::YELLOW));
        assert_eq!(
            (true, Some(Spot::YELLOW)),
            board.insert_top(3, Spot::YELLOW)
        );
    }

    #[test]
    fn edgecase_1() {
        let mut board = Board::new();

        assert_eq!((true, None), board.insert_top(6, Spot::RED));
        assert_eq!((true, None), board.insert_top(6, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(6, Spot::RED));
        assert_eq!((true, None), board.insert_top(6, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(5, Spot::RED));
        assert_eq!((true, None), board.insert_top(5, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(5, Spot::RED));
        assert_eq!((true, None), board.insert_top(5, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(5, Spot::RED));
        assert_eq!((true, None), board.insert_top(5, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::YELLOW));

        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
    }

    #[test]
    fn backward_diagonal_2() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(5, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(4, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(3, Spot::YELLOW));
        assert_eq!(
            (true, Some(Spot::YELLOW)),
            board.insert_top(2, Spot::YELLOW)
        );
    }

    #[test]
    fn backward_diagonal_3() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(6, Spot::RED));
        assert_eq!((true, None), board.insert_top(6, Spot::RED));
        assert_eq!((true, None), board.insert_top(5, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(5, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(5, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(4, Spot::RED));
        assert_eq!((true, None), board.insert_top(4, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(4, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(5, Spot::YELLOW));
        assert_eq!(
            (true, Some(Spot::YELLOW)),
            board.insert_top(6, Spot::YELLOW)
        );
    }

    #[test]
    fn vertical_1() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, Some(Spot::RED)), board.insert_top(0, Spot::RED));
    }

    #[test]
    fn vertical_2() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
    }

    #[test]
    fn vertical_3() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(0, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, Some(Spot::RED)), board.insert_top(0, Spot::RED));
    }

    #[test]
    fn horizontal_1() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(1, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::RED));
    }

    #[test]
    fn horizontal_2() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(1, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, Some(Spot::RED)), board.insert_top(3, Spot::RED));
    }

    #[test]
    fn horizontal_3() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(1, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, None), board.insert_top(3, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(1, Spot::RED));
        assert_eq!((true, None), board.insert_top(2, Spot::RED));
        assert_eq!((true, Some(Spot::RED)), board.insert_top(3, Spot::RED));
    }

    #[test]
    fn overflow_test() {
        let mut board = Board::new();
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((true, None), board.insert_top(0, Spot::YELLOW));
        assert_eq!((true, None), board.insert_top(0, Spot::RED));
        assert_eq!((false, None), board.insert_top(0, Spot::YELLOW));
    }
}
