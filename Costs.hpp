/*
 * Made by DanZimm on Sun Apr 2 17:15:26 EDT 2017
 */
#pragma once

#include "Activators.hpp"

struct MSEPrime;
struct MSE {
  using Prime = MSEPrime;
  inline double operator()(const Matrix& expected, const Matrix& activation) {
    return 0.5 * (expected - activation).normSquared();
  }
};

struct MSEPrime {
  template<typename Activator, typename ActivatorPrime = typename Activator::Prime>
  inline Matrix operator()(const Matrix& expected, 
                           const Matrix& activation,
                           const Matrix& z,
                           Activator act,
                           ActivatorPrime actPrime) {
    size_t height = expected.rows();
    Matrix delta = activation - expected;
    for (size_t j = 0; j < height; j++) {
      delta[j][0] = (activation[j][0] - expected[j][0])
          * actPrime(z[j][0], activation[j][0]);
    }
    return delta;
  }
};


struct CrossEntropyPrime;
struct CrossEntropy {
  using Prime = CrossEntropyPrime;
  inline double operator()(const Matrix& expected, const Matrix& activation) {
    double result = 0.0;
    size_t height = expected.rows();
    for (size_t i = 0; i < height; i++) {
      double currentExpected = expected[i][0];
      double currentActivation = activation[i][0];
      result += currentExpected * log(currentActivation)
            + (1 - currentExpected) * log(1 - currentActivation);
    }
    return -result;
  }
};

struct CrossEntropyPrime {
  template<typename Activator, typename ActivatorPrime = typename Activator::Prime>
  inline Matrix operator()(const Matrix& expected,
                           const Matrix& activation,
                           const Matrix& z,
                           Activator act,
                           ActivatorPrime actPrime) {
    size_t height = expected.rows();
    Matrix result(height, 1, Matrix::garbage);
    for (size_t j = 0; j < height; j++) {
      double currentAct = activation[j][0];
      double y = expected[j][0];
      double currentZ = z[j][0];
      result[j][0] = actPrime(currentZ, currentAct)
            * (currentAct - y) / (currentAct * (1.0 - currentAct));
    }
    return result;
  }
};

template<>
inline Matrix CrossEntropyPrime::operator()(const Matrix& expected,
                                            const Matrix& activation,
                                            const Matrix& z,
                                            Sigmoid act,
                                            SigmoidPrime actPrime) {
  return activation - expected;
}

