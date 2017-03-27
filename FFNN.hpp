/*
 * Made by DanZimm on Sun Mar 26 03:49:51 CDT 2017
 */
#pragma once

#include <array>

#include "Matrix.hpp"

template<size_t numberOfLayers>
struct FFNN {
  static_assert(numberOfLayers > 1, "A FFNN must have one layer");
 public:

  FFNN(std::array<unsigned, numberOfLayers>& layerSizes) {
       
  }

 private:
  std::array<Matrix, numberOfLayers - 1> _weights;
  std::array<std::vector<double>, numberOfLayers - 1> _biases;
};

