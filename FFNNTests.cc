/*
 * Made by DanZimm on Tue Mar 28 16:26:27 CDT 2017
 */

#include "FFNN.hpp"
#include "Test.hpp"

struct FFNNInitTest: Test {
  using Test::Test;
  virtual void run() {
    FFNN<2> nn(std::array<size_t, 2>{ {2, 4} });
    std::vector<std::pair<Matrix, Matrix>> trainingData;
    for (size_t i = 0; i < 500; i++) {
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(2, 1, { 0.0, 0.0 }),
            Matrix(4, 1, { 1.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(2, 1, { 0.0, 1.0 }),
            Matrix(4, 1, { 0.0, 2.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(2, 1, { 1.0, 0.0 }),
            Matrix(4, 1, { 0.0, 0.0, 1.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(2, 1, { 1.0, 1.0 }),
            Matrix(4, 1, { 0.0, 0.0, 0.0, 1.0 })));
    }
    for (size_t i = 0; i < 100; i++) {
      nn(trainingData);
    }
    std::cout << "00 -> " << std::to_string(nn(Matrix(2, 1, { 0.0, 0.0 }))) << std::endl;
    std::cout << "01 -> " << std::to_string(nn(Matrix(2, 1, { 0.0, 1.0 }))) << std::endl;
    std::cout << "10 -> " << std::to_string(nn(Matrix(2, 1, { 1.0, 0.0 }))) << std::endl;
    std::cout << "11 -> " << std::to_string(nn(Matrix(2, 1, { 1.0, 1.0 }))) << std::endl;
  }
};

struct FFNNTestSuite: TestSuite {
  FFNNTestSuite(const char* name) : TestSuite(name) {
    addTest(new FFNNInitTest("FFNNInit"));
  }
};

DeclareTest(FFNNTestSuite, FFNNTests)
