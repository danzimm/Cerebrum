/*
 * Made by DanZimm on Tue Mar 28 20:53:17 CDT 2017
 */
#pragma once

#include <cmath>

#include "Matrix.hpp"

struct SigmoidPrime;
struct Sigmoid {
  using Prime = SigmoidPrime;
  inline Matrix operator()(const Matrix& other) {
    return other.apply(Sigmoid::operation);
  }
  inline Matrix& operator()(Matrix& other) {
    return other.applyInPlace(Sigmoid::operation);
  }
  static inline double operation(double value) {
    return 1.0 / ( exp(-value) + 1.0 );
  }
};

struct SigmoidPrime {
  inline double operator()(const Matrix& other, size_t k, size_t j) {
    return k == j ? SigmoidPrime::operation(other[k][0]) : 0.0;
  }
  static inline double operation(double value) {
    return Sigmoid::operation(value) * ( 1.0 - Sigmoid::operation(value) );
  }
};

struct SoftMaxPrime;
struct SoftMax {
  using Prime = SoftMaxPrime;
  inline Matrix operator()(const Matrix& other) {
    Matrix result = other.apply(exp);
    result *= (1.0 / result.sum());
    return result;
  }
  inline Matrix& operator()(Matrix& other) {
    return other.applyInPlace(exp) *= (1.0 / other.sum());
  }
};

struct SoftMaxPrime {
  inline double operator()(const Matrix& other, size_t k, size_t j) {
    Matrix tmp = SoftMax()(other);
    return tmp[k][0] * ( ( j == k ? 1.0 : 0.0 ) - tmp[j][0] );
  }
};

