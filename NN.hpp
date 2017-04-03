/*
 * Made by DanZimm on Sun Mar 26 03:49:51 CDT 2017
 */
#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <iterator>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "Activators.hpp"
#include "Matrix.hpp"

template<size_t numberOfLayers, typename Activator=Sigmoid, typename ActivatorPrime=typename Activator::Prime>
struct NN {
  static_assert(numberOfLayers > 1, "A NN must have one layer");
 public:
  template<typename RandDist = std::uniform_real_distribution<>>
  NN(const std::array<size_t, numberOfLayers>& layerSizes, 
     RandDist dist = std::uniform_real_distribution<>(0.0, 1.0)) : _learningRate(0.1), _miniBatchSize(32), _verbosity(0), _concurrent(false) {
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      size_t size = layerSizes[i];
      size_t nextSize = layerSizes[i + 1];
      _weights[i] = Matrix(nextSize, size, gen, dist);
      _biases[i] = Matrix(nextSize, 1, gen, dist);
    }
  }

  Matrix operator()(const Matrix& inputs) {
    Matrix result(inputs);
    Activator act;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      // i = `currentLayer - 1` where current layer is the
      // layer being simulated
      result = _weights[i] * result;
      result += _biases[i];
      act(result);
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
    auto end = data.end();

    std::array<size_t, numberOfLayers> sizes;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      sizes[i] = _weights[i].columns();
    }
    sizes[numberOfLayers - 1] = _weights.back().rows();
    for (auto iter = data.begin(); iter < end; iter += _miniBatchSize) {
      auto last = iter + _miniBatchSize;
      if (last > end) {
        last = end;
      }
      _trainWithBatch(iter, last, sizes);
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

  void setVerbosity(unsigned verb) {
    _verbosity = verb;
  }

  unsigned verbosity() const {
    return _verbosity;
  }

  void setConcurrent(bool concur) {
    _concurrent = concur;
  }

  bool concurrent() const {
    return _concurrent;
  }
 private:
  template<typename Iter>
  void _trainWithBatch(Iter begin,
                       Iter end,
                       const std::array<size_t, numberOfLayers>& sizes) {
    size_t size = end - begin;
    std::array<Matrix, numberOfLayers - 1> deltaWeights;
    std::array<Matrix, numberOfLayers - 1> deltaBiases;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      size_t size = sizes[i];
      size_t nextSize = sizes[i + 1];
      deltaWeights[i] = Matrix(nextSize, size);
      deltaBiases[i] = Matrix(nextSize, 1);
    }

    unsigned workerCount = std::thread::hardware_concurrency();
    if (_concurrent && size > workerCount) {
      std::mutex iterLock;
      std::vector<std::thread> threads;
      for (size_t i = 0; i < workerCount; i++) {
        threads.push_back(
            std::thread(NN<numberOfLayers, Activator, ActivatorPrime>
                          ::template _backPropgateFromQueue<Iter>,
                        std::ref(begin),
                        end,
                        std::ref(iterLock),
                        std::ref(_weights),
                        std::ref(_biases),
                        std::ref(deltaWeights),
                        std::ref(deltaBiases)));
      }
      for (size_t i = 0; i < workerCount; i++) {
        threads[i].join();
      }
    } else {
      std::for_each(begin, end, [&](const std::pair<Matrix, Matrix>* pair) {
          NN::_backPropagate(*pair, _weights, _biases, deltaWeights, deltaBiases);
      });
    }
    double multi = _learningRate / (double)size;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      deltaWeights[i] *= -multi;
      deltaBiases[i] *= -multi;
      _weights[i] += deltaWeights[i];
      _biases[i] += deltaBiases[i];
    }
  }

  template<typename Iter>
  static void _backPropgateFromQueue(Iter& begin,
                                     Iter end,
                                     std::mutex& iterLock,
                                     const std::array<Matrix, numberOfLayers - 1>& weights,
                                     const std::array<Matrix, numberOfLayers - 1>& biases,
                                     std::array<Matrix, numberOfLayers - 1>& deltaWeights,
                                     std::array<Matrix, numberOfLayers - 1>& deltaBiases) {
    while (true) {
      iterLock.lock();
      if (begin == end) {
        iterLock.unlock();
        break;
      }
      auto pair = *begin++;
      iterLock.unlock();
      _backPropagate(*pair, weights, biases, deltaWeights, deltaBiases);
    }
  }

  static void _backPropagate(const std::pair<Matrix, Matrix>& data,
                             const std::array<Matrix, numberOfLayers - 1>& weights,
                             const std::array<Matrix, numberOfLayers - 1>& biases,
                             std::array<Matrix, numberOfLayers - 1>& deltaWeights,
                             std::array<Matrix, numberOfLayers - 1>& deltaBiases) {
    std::array<Matrix, numberOfLayers - 1> z;
    std::array<Matrix, numberOfLayers> a;
    a[0] = data.first;
    Activator act;
    ActivatorPrime actPrime;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      Matrix& currentZ = z[i];
      currentZ = weights[i] * a[i];
      currentZ += biases[i];
      a[i + 1] = act(static_cast<std::add_const_t<decltype(currentZ)>>(currentZ));
    }
    // TODO: make cost function a parameter
    Matrix current(a.back());
    current -= data.second;
    const auto& lastZ = z.back();
    Matrix delta(current.rows(), current.columns(), Matrix::garbage);
    const size_t lastHeight = lastZ.rows();
    const auto& lastActivation = a.back();
    for (size_t j = 0; j < lastHeight; j++) {
      double sum = 0;
      for (size_t k = 0; k < lastHeight; k++) {
        sum += current[k][0] * actPrime(lastZ, lastActivation, k, j);
      }
      delta[j][0] = sum;
    }
    deltaBiases.back() += delta;
    deltaWeights.back() += delta * a[numberOfLayers - 2].transpose();
    // we want to start at the second to last layer of weights/biases so index
    // is numberOfLayers - 2 - 1 = numberOfLayers - 3 but we need to add 1
    // because our counter is 1 above the desired index cuz unsigned.
    for (size_t l = numberOfLayers - 2; l > 0; l--) {
      // actual layer we're on
      size_t layer = l - 1;
      const auto& currentZ = z[layer];
      const auto& currentActivation = a[l];
      const size_t height = currentZ.rows();

      current = weights[l].transpose() * delta;
      delta = Matrix(current.rows(), current.columns(), Matrix::garbage);
      for (size_t j = 0; j < height; j++) {
        double sum = 0;
        for (size_t k = 0; k < height; k++) {
          sum += current[k][0] * actPrime(currentZ, currentActivation, k, j);
        }
        delta[j][0] = sum;
      }

      deltaBiases[layer] += delta;
      deltaWeights[layer] += delta * a[layer].transpose();
    }
  }

  std::array<Matrix, numberOfLayers - 1> _weights;
  std::array<Matrix, numberOfLayers - 1> _biases;
  double _learningRate;
  size_t _miniBatchSize;
  unsigned _verbosity;
  bool _concurrent;
  std::mutex _lock;
};

