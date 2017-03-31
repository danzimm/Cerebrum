/*
 * Made by DanZimm on Tue Mar 28 20:53:17 CDT 2017
 */
#pragma once

#include <cmath>

#include "Matrix.hpp"

struct Sigmoid {
  Matrix operator()(const Matrix& other) {
    return other.apply(Sigmoid::operation);
  }
  Matrix& operator()(Matrix& other) {
    return other.applyInPlace(Sigmoid::operation);
  }
  static double operation(double value) {
    return 1.0 / ( exp(-value) + 1.0 );
  }
};

struct SigmoidPrime {
  Matrix operator()(const Matrix& other) {
    return other.apply(SigmoidPrime::operation);
  }
  Matrix& operator()(Matrix& other) {
    return other.applyInPlace(SigmoidPrime::operation);
  }
  static double operation(double value) {
    return Sigmoid::operation(value) * ( 1.0 - Sigmoid::operation(value) );
  }
};
