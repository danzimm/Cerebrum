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

template<size_t numberOfLayers, typename Activator=Sigmoid, typename ActivatorPrime=SigmoidPrime>
struct NN {
  static_assert(numberOfLayers > 1, "A NN must have one layer");
 public:
  NN(const std::array<size_t, numberOfLayers>& layerSizes) : _learningRate(0.1), _miniBatchSize(32), _verbosity(0), _concurrent(false) {
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

    std::mutex iterLock;

    unsigned workerCount = std::thread::hardware_concurrency();
    if (_concurrent && size > workerCount) {
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
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      Matrix& currentZ = z[i];
      currentZ = weights[i] * a[i];
      currentZ += biases[i];
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
      current = weights[i].transpose() * current;
      current.elementWiseProductInPlace(z[layer].apply(ActivatorPrime()));
      deltaBiases[layer] += current;
      deltaWeights[layer] += current * a[layer].transpose();
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

