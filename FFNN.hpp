/*
 * Made by DanZimm on Sun Mar 26 03:49:51 CDT 2017
 */
#pragma once

#include <algorithm>
#include <array>
#include <iterator>
#include <random>
#include <vector>

#include "Activators.hpp"
#include "Matrix.hpp"

template<size_t numberOfLayers, typename Activator=Sigmoid, typename ActivatorPrime=SigmoidPrime>
struct FFNN {
  static_assert(numberOfLayers > 1, "A FFNN must have one layer");
 public:
  FFNN(const std::array<size_t, numberOfLayers>& layerSizes) : _learningRate(0.1), _miniBatchSize(4) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 1.1);
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      size_t size = layerSizes[i];
      size_t nextSize = layerSizes[i + 1];
      _weights[i] = Matrix(nextSize, size, gen, dis);
      _biases[i] = Matrix(nextSize, 1, gen, dis);
    }
  }

  Matrix operator()(const Matrix& inputs) {
    Matrix result(inputs);
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      // i = `currentLayer - 1` where current layer is the
      // layer being simulated
      result = _weights[i] * result;
      result += _biases[i];
      result.applyInPlace(Activator());
    }
    return result;
  }

  void operator()(const std::vector<std::pair<Matrix, Matrix>>& trainingCases) {
    size_t size = trainingCases.size();
    std::vector<const std::pair<Matrix, Matrix>*> data(size);
    for (size_t i = 0; i < size; i++) {
      data[i] = &trainingCases[i];
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    for (auto iter = data.begin(); iter < data.end(); iter += _miniBatchSize) {
      auto last = iter + _miniBatchSize;
      _trainWithBatch(iter, last);
    }
  }

  void setLearningRate(double rate) {
    _learningRate = rate;
  }

  double learningRate() const {
    return _learningRate;
  }

  void setMiniBatchSize(size_t size) {
    _miniBatchSize = size;
  }

  size_t miniMatchSize() const {
    return _miniBatchSize;
  }
 private:

  template<typename Iter>
  void _trainWithBatch(Iter begin, Iter end) {
    size_t size = end - begin;
    std::array<Matrix, numberOfLayers - 1> deltaWeights;
    std::array<Matrix, numberOfLayers - 1> deltaBiases;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      auto& weightMat = _weights[i];
      auto& biasMat = _biases[i];
      deltaWeights[i] = Matrix(weightMat.rows(), weightMat.columns());
      deltaBiases[i] = Matrix(biasMat.rows(), biasMat.columns());
    }
    std::for_each(begin, end, [&](const std::pair<Matrix, Matrix>* pair) {
      _backPropagate(*pair, deltaWeights, deltaBiases);
    });
    double multi = _learningRate / (double)size;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      _weights[i] += (deltaWeights[i] *= -multi);
      _biases[i] += (deltaBiases[i] *= -multi);
    }
  }

  void _backPropagate(const std::pair<Matrix, Matrix>& data,
                      std::array<Matrix, numberOfLayers - 1>& deltaWeights,
                      std::array<Matrix, numberOfLayers - 1>& deltaBiases) {
    std::array<Matrix, numberOfLayers - 1> z;
    std::array<Matrix, numberOfLayers> a;
    a[0] = data.first;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      Matrix& currentZ = z[i];
      currentZ = _weights[i] * a[i];
      currentZ += _biases[i];
      a[i + 1] = currentZ.apply(Activator());
    }
    Matrix current(a.back());
    current -= data.second;
    current.elementWiseProductInPlace(z.back().apply(ActivatorPrime()));
    deltaBiases.back() += current;
    deltaWeights.back() += current * a[numberOfLayers - 2].transpose();
    // we want to start at the second to last layer of weights/biases so index
    // is numberOfLayers - 2 - 1 = numberOfLayers - 3 but we need to add 1
    // because our counter is 1 above the desired index cuz unsigned.
    for (size_t i = numberOfLayers - 2; i > 0; i--) {
      // actual layer we're on
      size_t layer = i - 1;
      current = _weights[i].transpose() * current;
      current.elementWiseProductInPlace(z[layer].apply(ActivatorPrime()));
      deltaBiases[layer] += current;
      deltaWeights[layer] += current * a[layer].transpose();
    }
  }

  std::array<Matrix, numberOfLayers - 1> _weights;
  std::array<Matrix, numberOfLayers - 1> _biases;
  double _learningRate;
  size_t _miniBatchSize;
};

