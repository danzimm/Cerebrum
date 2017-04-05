/*
 * Made by DanZimm on Sun Apr 2 17:15:26 EDT 2017
 */
#pragma once

struct MSEPrime;
struct MSE {
  using Prime = MSEPrime;
  inline double operator()(const Matrix& expected, const Matrix& actual) {
    return 0.5 * (expected - actual).normSquared();
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
    Matrix delta(height, 1, Matrix::garbage);
    for (size_t j = 0; j < height; j++) {
      double sum = 0;
      for (size_t k = 0; k < height; k++) {
        sum += (activation[k][0] - expected[k][0]) * actPrime(z, activation, k, j);
      }
      delta[j][0] = sum;
    }
    return delta;
  }
};


struct CrossEntropyPrime;
struct CrossEntropy {
  using Prime = CrossEntropyPrime;
  inline double operator()(const Matrix& expected, const Matrix& actual) {
    double result = 0.0;
    size_t height = expected.rows();
    for (size_t i = 0; i < height; i++) {
      double currentExpected = expected[i][0];
      double currentActual = actual[i][0];
      result += currentExpected * log(currentActual) + (1 - currentExpected) * log(1 - currentActual);
    }
    return -result;
  }
};

struct CrossEntropyPrime {
  template<typename Activator, typename ActivatorPrime = typename Activator::Prime>
  inline Matrix operator()(const Matrix& expected,
                           const Matrix& actual,
                           const Matrix& z,
                           Activator act,
                           ActivatorPrime actPrime) {
    return actual - expected;
  }
};

