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
#include "Costs.hpp"
#include "Matrix.hpp"

template<size_t numberOfLayers,
         typename Cost=MSE,
         typename Activator=Sigmoid,
         typename CostPrime=typename Cost::Prime,
         typename ActivatorPrime=typename Activator::Prime>
struct NN {
  static_assert(numberOfLayers > 1, "A NN must have one layer");
 public:
  template<typename RandDist = std::uniform_real_distribution<>>
  NN(const std::array<size_t, numberOfLayers>& layerSizes, 
     RandDist dist = std::uniform_real_distribution<>(0.0, 1.0)) 
        : _learningRate(0.1), 
          _miniBatchSize(32), _verbosity(0), _concurrent(false) {
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

  void train(const std::vector<std::pair<Matrix, Matrix>>& trainingCases) {
    // We want to be stochastic, so copy the array and then shuffle it
    size_t size = trainingCases.size();
    std::vector<const std::pair<Matrix, Matrix>*> data(size);
    for (size_t i = 0; i < size; i++) {
      data[i] = &trainingCases[i];
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
    auto end = data.end();

    // Cache the layer sizes
    std::array<size_t, numberOfLayers> sizes;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      sizes[i] = _weights[i].columns();
    }
    sizes[numberOfLayers - 1] = _weights.back().rows();
    // Actual pick out a minibatch and train with it
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
    // We need a cache for the \partial W_{jk}, \partial b_j
    // so we init an array of matrices, each with the proper size
    size_t batchSize = end - begin;
    std::array<Matrix, numberOfLayers - 1> deltaWeights;
    std::array<Matrix, numberOfLayers - 1> deltaBiases;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      size_t size = sizes[i];
      size_t nextSize = sizes[i + 1];
      deltaWeights[i] = Matrix(nextSize, size);
      deltaBiases[i] = Matrix(nextSize, 1);
    }

    // Check if we should work concurrently - this is beta atm
    // as it doesn't currently give any speed up due to threads
    // often being created/destroyed
    unsigned workerCount = std::thread::hardware_concurrency();
    if (_concurrent && batchSize > workerCount) {
      std::mutex iterLock;
      std::vector<std::thread> threads;
      for (size_t i = 0; i < workerCount; i++) {
        threads.push_back(
            std::thread(NN<numberOfLayers, 
                           Cost, Activator, CostPrime, ActivatorPrime>
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
      // For each input/output in this batch do the back propogation
      std::for_each(begin, end, [&](const std::pair<Matrix, Matrix>* pair) {
        NN::_backPropagate(*pair, _weights, _biases, deltaWeights, deltaBiases);
      });
    }
    // The back propogation doesn't apply the learning rate or divide by the
    // number of samples being trained with, so do that now
    double multi = _learningRate / (double)batchSize;
    // We now have calculated the total sum of the \partial W_{jk}, \partial b
    // so add it to the actual weights
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      _weights[i] += (deltaWeights[i] *= -multi);
      _biases[i] += (deltaBiases[i] *= -multi);
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
    // Simple routine to pop from a thread safe queue the sample that should be
    // back propogated
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
    // Instantiate the activators/costs
    Activator act;
    ActivatorPrime actPrime;
    CostPrime cost;
    // Set up array for calculating weighted inputs/activations
    // for use later
    std::array<Matrix, numberOfLayers - 1> z;
    std::array<Matrix, numberOfLayers> a;
    a[0] = data.first;
    for (size_t i = 0; i < numberOfLayers - 1; i++) {
      Matrix& currentZ = z[i];
      currentZ = weights[i] * a[i];
      currentZ += biases[i];
      a[i + 1] = act(static_cast<std::add_const_t<decltype(currentZ)>>(currentZ));
    }

    // 1) Calculate gradient descent for last layer:
    // 1.1) Calculate \delta^L_j = \frac{\partial C}{\partial z^L_j}
    //      for every j in the output layer
    const auto& lastZ = z.back();
    Matrix delta = cost(data.second, a.back(), lastZ, act, actPrime);
    // 1.2) Calculate \frac{\partial C}{\partial b^L_j} = \delta^L_j
    deltaBiases.back() += delta;
    // 1.3) Calculate \frac{\partial C}{\partial w^L_{jk}} = \delta^L_j a^{L-1}_{k}
    deltaWeights.back() += delta * a[numberOfLayers - 2].transpose();
    // we want to start at the second to last layer of weights/biases so index
    // is numberOfLayers - 2 - 1 = numberOfLayers - 3 but we need to add 1
    // because our counter is 1 above the desired index cuz unsigned.
    for (size_t l = numberOfLayers - 2; l > 0; l--) {
      // actual layer we're on
      size_t layer = l - 1;
      const auto& currentZ = z[layer];
      // note that a has an additional element at the beginning being the
      // input, so for a l = layer, layer = l - 1, in other words a[l] is
      // the a for the current layer
      const auto& currentActivation = a[l];
      const size_t height = currentZ.rows();
      // 2) Use gradient descent trick to calculate lower layers iteratively
      // 2.1) Calculate \delta^l = (w^{l+1})^T * \delta^{l+1} * \sigma'(z^l)
      //      (note that here l = layer and l+1 = layer = 1 = `l`; `l` is the l
      //       in code)
      delta = weights[l].transpose() * delta;
      for (size_t j = 0; j < height; j++) {
        delta[j][0] *= actPrime(currentZ[j][0], currentActivation[j][0]);
      }
      // 2.2) Calculate \frac{\partial C}{\partial b^l_j} = \delta^l_j
      deltaBiases[layer] += delta;
      // 2.3) Calculate \frac{\partial C}{\partial w^l_{jk}} = \delta^l_j a^{l-1}_k
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

