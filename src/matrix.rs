use libc::c_int;
use rblas::attribute::Transpose;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};

impl<T> rblas::Matrix<T> for Matrix<T>
where
    T: Add<Output = T>,
{
    #[inline]
    fn rows(&self) -> c_int {
        self.rows as c_int
    }

    #[inline]
    fn cols(&self) -> c_int {
        self.cols as c_int
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.values.as_ptr()
    }

    #[inline]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.values.as_mut_ptr()
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct Matrix<T>
where
    T: Add<Output = T>,
{
    pub values: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T> Matrix<T>
where
    T: Add<Output = T> + std::ops::AddAssign + Default + Clone,
{
    pub fn from(vector: Vec<T>, rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            values: vector,
        }
    }

    pub fn into_row(vector: Vec<T>) -> Self {
        Matrix {
            rows: vector.len(),
            cols: 1,
            values: vector,
        }
    }

    #[inline]
    pub fn from_rand(rows: usize, columns: usize, rand_fn: &mut dyn FnMut() -> T) -> Self {
        let mut values: Vec<T> = Vec::with_capacity(rows * columns);

        for _ in 0..(rows * columns) {
            values.push(rand_fn());
        }

        Matrix {
            rows,
            cols: columns,
            values,
        }
    }

    #[inline]
    pub fn alloca(rows: usize, columns: usize) -> Self {
        let values = vec![Default::default(); columns * rows];
        Matrix {
            rows,
            cols: columns,
            values,
        }
    }

    #[allow(non_snake_case)]
    #[inline]
    pub fn T(self) -> Self {
        self.transpose()
    }

    #[inline]
    fn transpose(self) -> Self {
        let mut c = Self::alloca(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let c_idx = c.cidx(j, i);
                c.values[c_idx] = self.values[self.cidx(i, j)].clone();
            }
        }
        c
    }

    pub fn map(&mut self, func: &mut dyn FnMut(T) -> T) {
        for item in self.values.iter_mut() {
            *item = func(item.clone())
        }
    }

    pub fn map_enumerate(&mut self, func: &mut dyn FnMut(usize, usize, T) -> T) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let idx = self.cidx(i, j);
                self.values[idx] = func(i, j, self.values[idx].clone())
            }
        }
    }

    #[inline]
    pub fn cidx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    #[inline]
    pub fn push(&mut self, values: &mut Vec<T>) {
        debug_assert_eq!(values.len(), self.cols);

        self.rows += 1;
        self.values.append(values);
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.values[self.cidx(row, col)].clone()
    }
}

impl<T> Add<Matrix<T>> for Matrix<T>
where
    T: Add<Output = T> + std::ops::AddAssign,
{
    type Output = Matrix<T>;

    #[inline]
    fn add(mut self, other: Matrix<T>) -> Matrix<T> {
        debug_assert_eq!(self.values.len(), other.values.len());
        for (i, other) in other.values.into_iter().enumerate() {
            self.values[i] += other;
        }

        self
    }
}

impl<T> Add<T> for Matrix<T>
where
    T: Add<Output = T> + std::ops::AddAssign + Clone,
{
    type Output = Matrix<T>;

    #[inline]
    fn add(mut self, other: T) -> Matrix<T> {
        for i in 0..(self.rows * self.cols) {
            self.values[i] += other.clone();
        }

        self
    }
}

pub trait Bound {
    fn upper() -> Self;
    fn lower() -> Self;
}

impl Bound for f32 {
    fn upper() -> Self {
        1.0
    }
    fn lower() -> Self {
        0.0
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
        + std::ops::AddAssign
        + Bound
        + rblas::Gemm,
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
    #[inline]
    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        // m has to be equal to m
        debug_assert_eq!(self.cols, other.rows);
        let n = self.rows;
        let p = other.cols;
        let mut target = Matrix::alloca(n, p);
        rblas::Gemm::gemm(
            &T::upper(),
            Transpose::NoTrans,
            &self,
            Transpose::NoTrans,
            &other,
            &T::lower(),
            &mut target,
        );
        target
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
        + std::ops::AddAssign
        + Bound
        + rblas::Gemm,
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
    #[inline]
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        // m has to be equal to m
        debug_assert_eq!(self.cols, other.rows);
        let n = self.rows;
        let p = other.cols;
        let mut target = Matrix::alloca(n, p);
        rblas::Gemm::gemm(
            &T::upper(),
            Transpose::NoTrans,
            self,
            Transpose::NoTrans,
            other,
            &T::lower(),
            &mut target,
        );
        target
    }
}

impl<T> Mul<T> for Matrix<T>
where
    T: Mul<Output = T> + std::ops::MulAssign + Clone + std::ops::Add<Output = T>,
{
    type Output = Matrix<T>;

    #[inline]
    fn mul(mut self, other: T) -> Matrix<T> {
        for i in 0..(self.rows * self.cols) {
            self.values[i] *= other.clone();
        }

        self
    }
}

#[macro_export]
macro_rules! mat {
    ($($($e: expr),+);*) => {{
        let mut vec = Vec::new();
        let mut total = 0;
        let mut rows = 0;
        $(
            $(
                vec.push($e);
                total += 1;
            )+
            rows += 1;
        )*

        Matrix::from(vec, rows, total / rows)
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
        let first_mat = mat![1.0, 2.0, 3.0; 1.0, 2.0, 3.0; 1.0, 2.0, 3.0];

        // 3 by 1
        let second_mat = mat![2.0; 10.0; 3.0];
        assert_eq!(first_mat * second_mat, mat![31.0; 31.0; 31.0]);
    }

    #[test]
    fn mul_matrices_2() {
        // 2 by 2
        let first_mat = mat![1.0, 2.0; 2.0, 1.0];

        // 2 by 2
        let second_mat = mat![3.0, 1.0; 1.0, 3.0];
        assert_eq!(first_mat * second_mat, mat![5.0, 7.0; 7.0, 5.0]);
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
