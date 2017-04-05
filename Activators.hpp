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
  inline double operator()(const Matrix& other, const Matrix& sigma, size_t k, size_t j) {
    if (k != j) {
      return 0.0;
    }
    double sig = sigma[k][0];
    return sig * (1.0 - sig);
  }
};

struct SoftMaxPrime;
struct SoftMax {
  using Prime = SoftMaxPrime;
  inline Matrix operator()(const Matrix& other) {
    Matrix result = other - other.max();
    result.applyInPlace(exp);
    result *= (1.0 / result.sum());
    return result;
  }
  inline Matrix& operator()(Matrix& other) {
    other -= other.max();
    other.applyInPlace(exp);
    other *= (1.0 / other.sum());
    return other;
  }
};

struct SoftMaxPrime {
  inline double operator()(const Matrix& other, const Matrix& softMax, size_t k, size_t j) {
    return softMax[k][0] * ( (j == k ? 1.0 : 0.0) - softMax[j][0] );
  }
};

