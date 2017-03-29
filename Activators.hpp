/*
 * Made by DanZimm on Tue Mar 28 20:53:17 CDT 2017
 */
#pragma once

#include <cmath>

struct Sigmoid {
  double operator()(double value) {
    return 1.0 / (1.0 + exp(-value));
  }
};

struct SigmoidPrime {
  double operator()(double value) {
    auto s = Sigmoid();
    return s(value) * ( 1.0 - s(value) );
  }
};
