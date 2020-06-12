use rand::Rng;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Matrix<T>
where
    T: Add<Output = T>,
{
    pub values: Vec<Vec<T>>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Matrix<T>
where
    T: Add<Output = T> + std::ops::AddAssign + Default + Clone,
{
    pub fn from(vector: Vec<Vec<T>>) -> Self {
        Matrix {
            rows: vector.len(),
            cols: vector.first().unwrap_or(&vec![]).len(),
            values: vector,
        }
    }

    pub fn into_row(vector: Vec<T>) -> Self {
        Matrix {
            rows: vector.len(),
            cols: 1,
            values: Matrix::from(vec![vector]).T().values,
        }
    }

    pub fn from_rand(rows: usize, columns: usize, rand_fn: &mut dyn FnMut() -> T) -> Self {
        let mut values: Vec<Vec<T>> = Vec::with_capacity(rows);

        for _ in 0..rows {
            let mut cols: Vec<T> = Vec::with_capacity(columns);
            for _ in 0..columns {
                cols.push(rand_fn());
            }
            values.push(cols);
        }

        Matrix {
            rows,
            cols: columns,
            values,
        }
    }

    pub fn alloca(rows: usize, columns: usize) -> Self {
        let cols = vec![Default::default(); columns];
        let values = vec![cols; rows];
        Matrix {
            rows,
            cols: columns,
            values,
        }
    }

    #[allow(non_snake_case)]
    pub fn T(self) -> Self {
        self.transpose()
    }

    fn transpose(self) -> Self {
        let mut c = Self::alloca(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                c.values[j][i] = self.values[i][j].clone();
            }
        }
        c
    }

    pub fn mapped<C: Fn(T) -> T + Sync>(self, func: &C) -> Self {
        Matrix {
            cols: self.cols,
            rows: self.rows,
            values: self
                .values
                .into_iter()
                .map(|row| row.into_iter().map(|x| func(x)).collect::<Vec<_>>())
                .collect(),
        }
    }

    pub fn map(&mut self, func: &mut dyn FnMut(T) -> T) {
        self.values = self
            .values
            .iter()
            .map(|row| row.into_iter().map(|x| func(x.clone())).collect::<Vec<_>>())
            .collect::<Vec<_>>();
    }

    pub fn map_enumerate(&mut self, func: &mut dyn FnMut(usize, usize, T) -> T) {
        self.values = self
            .values
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.into_iter()
                    .enumerate()
                    .map(|(j, x)| func(i, j, x.clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
    }

    pub fn push(&mut self, value: Vec<T>) {
        assert_eq!(value.len(), self.cols);

        self.rows += 1;
        self.values.push(value);
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        self.values[row][col].clone()
    }
}

impl<T> Add<Matrix<T>> for Matrix<T>
where
    T: Add<Output = T> + std::ops::AddAssign,
{
    type Output = Matrix<T>;

    fn add(mut self, other: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.values.len(), other.values.len());
        for (i, value) in other.values.into_iter().enumerate() {
            for (j, actual_val) in value.into_iter().enumerate() {
                self.values[i][j] += actual_val;
            }
        }

        self
    }
}

impl<T> Add<T> for Matrix<T>
where
    T: Add<Output = T> + std::ops::AddAssign + Clone,
{
    type Output = Matrix<T>;

    fn add(mut self, other: T) -> Matrix<T> {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.values[i][j] += other.clone();
            }
        }

        self
    }
}

impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Mul<Output = T>
        + std::ops::MulAssign
        + std::ops::Add<Output = T>
        + Default
        + Clone
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    type Output = Matrix<T>;

    /*
    Input: matrices A and B
        Let C be a new matrix of the appropriate size
        For i from 1 to n:
            For j from 1 to p:
                Let sum = 0
                For k from 1 to m:
                    Set sum ← sum + Aik × Bkj
                Set Cij ← sum
    Return C
    */
    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        // m has to be equal to m
        assert_eq!(self.cols, other.rows);

        let a = self;
        let b = other;

        // n = self.rows
        // m = self.cols OR other.rows
        // p = other.cols
        // Defined for clarity
        let n = a.rows;
        let m = a.cols;
        let p = b.cols;

        // Allocate output array size
        let mut c = Self::alloca(a.cols, b.cols);

        for i in 0..n {
            for j in 0..p {
                let mut sum: T = Default::default();
                for k in 0..m {
                    sum += a.values[i][k].clone() * b.values[k][j].clone()
                }
                c.values[i][j] = sum;
            }
        }

        c
    }
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Mul<Output = T>
        + std::ops::MulAssign
        + std::ops::Add<Output = T>
        + Default
        + Clone
        + std::fmt::Debug
        + std::ops::AddAssign,
{
    type Output = Matrix<T>;

    /*
    Input: matrices A and B
        Let C be a new matrix of the appropriate size
        For i from 1 to n:
            For j from 1 to p:
                Let sum = 0
                For k from 1 to m:
                    Set sum ← sum + Aik × Bkj
                Set Cij ← sum
    Return C
    */
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        // m has to be equal to m
        assert_eq!(self.cols, other.rows);

        let a = self;
        let b = other;

        // n = self.rows
        // m = self.cols OR other.rows
        // p = other.cols
        // Defined for clarity
        let n = a.rows;
        let m = a.cols;
        let p = b.cols;

        // Allocate output array size
        let mut c = Matrix::alloca(n, p);

        for i in 0..n {
            for j in 0..p {
                let mut sum: T = Default::default();
                for k in 0..m {
                    sum += a.values[i][k].clone() * b.values[k][j].clone()
                }
                c.values[i][j] = sum;
            }
        }

        c
    }
}

impl<T> Mul<T> for Matrix<T>
where
    T: Mul<Output = T> + std::ops::MulAssign + Clone + std::ops::Add<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(mut self, other: T) -> Matrix<T> {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.values[i][j] *= other.clone();
            }
        }

        self
    }
}

#[macro_export]
macro_rules! mat {
    ($($($e: expr),+);*) => {{
        let mut vec = Vec::new();
        $(
            let mut row = Vec::new();
            $(
                row.push($e);
            )+
            vec.push(row);
        )*

        Matrix::from(vec)
    }};
}

#[cfg(test)]
mod matrix_tests {
    use super::*;

    #[test]
    fn add_matrices() {
        let first_mat = mat![1, 2, 3; 1, 2, 3];
        let second_mat = mat![13, 20, 2; 13, 23, 33];
        assert_eq!(first_mat + second_mat, mat![14, 22, 5; 14, 25, 36]);
    }

    #[test]
    fn add_matrix_scalar() {
        let first_mat = mat![1, 2, 3];
        assert_eq!(first_mat + 2, mat![3, 4, 5]);
    }

    #[test]
    fn mul_matrices_1() {
        // 3 by 3
        let first_mat = mat![1, 2, 3; 1, 2, 3; 1, 2, 3];

        // 3 by 1
        let second_mat = mat![2; 10; 3];
        assert_eq!(first_mat * second_mat, mat![31; 31; 31]);
    }

    #[test]
    fn mul_matrices_2() {
        // 2 by 2
        let first_mat = mat![1, 2; 2, 1];

        // 2 by 2
        let second_mat = mat![3, 1; 1, 3];
        assert_eq!(first_mat * second_mat, mat![5, 7; 7, 5]);
    }

    #[test]
    fn mul_matrix_scalar() {
        let first_mat = mat![1, 2, 3];
        assert_eq!(first_mat * 2, mat![2, 4, 6]);
    }

    #[test]
    fn transpose_1() {
        let mat = mat![1; 2; 3];
        assert_eq!(mat.T(), mat![1, 2, 3]);
    }

    #[test]
    fn transpose_2() {
        let mat = mat![1, 2; 3, 4];
        assert_eq!(mat.T(), mat![1, 3; 2, 4]);
    }

    #[test]
    fn transpose_3() {
        let mat = mat![1, 2];
        assert_eq!(mat.T(), mat![1; 2]);
    }

    #[test]
    fn transpose_4() {
        let mat = mat![1, 2; 3, 4; 5, 6];
        assert_eq!(mat.T(), mat![1, 3, 5; 2, 4, 6]);
    }
}
