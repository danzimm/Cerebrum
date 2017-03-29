/*
 * Made by DanZimm on Tue Mar 28 16:26:27 CDT 2017
 */

#include "NN.hpp"
#include "Test.hpp"

#include <set>

struct NNTest: Test {
  using Test::Test;
  void ensureActivations(const Matrix& m, const std::set<size_t>& activations, double bound=0.9) {
    auto end = activations.end();
    for (size_t i = 0; i < m.rows(); i++) {
      if (activations.find(i) != end) {
        EnsureGreaterThan(m[i][0], bound, "Proper activation");
      } else {
        EnsureLessThan(m[i][0], 1-bound, "Proper lack of activation");
      }
    }
  }
};

struct NNBinToDecTest: NNTest {
  using NNTest::NNTest;
  virtual void run() {
    NN<2> nn(std::array<size_t, 2>{ { 4, 10 } });
    std::vector<std::pair<Matrix, Matrix>> trainingData;
    for (size_t i = 0; i < 1000; i++) {
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 0.0, 0.0, 0.0, 0.0 }),
            Matrix(10, 1, { 1.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 0.0, 0.0, 0.0, 1.0 }),
            Matrix(10, 1, { 0.0, 1.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 0.0, 0.0, 1.0, 0.0 }),
            Matrix(10, 1, { 0.0, 0.0, 1.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 0.0, 0.0, 1.0, 1.0 }),
            Matrix(10, 1, { 0.0, 0.0, 0.0, 1.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 0.0, 1.0, 0.0, 0.0 }),
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 1.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 0.0, 1.0, 0.0, 1.0 }),
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            1.0, 0.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 0.0, 1.0, 1.0, 0.0 }),
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 1.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 0.0, 1.0, 1.0, 1.0 }),
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 1.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 1.0, 0.0, 0.0, 0.0 }),
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 1.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(4, 1, { 1.0, 0.0, 0.0, 1.0 }),
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 1.0 })));
    }
    nn.setMiniBatchSize(10);
    for (size_t i = 0; i < 200; i++) {
      nn(trainingData);
    }
    ensureActivations(nn(Matrix(4, 1, { 0.0, 0.0, 0.0, 0.0 })), { 0 });
    ensureActivations(nn(Matrix(4, 1, { 0.0, 0.0, 0.0, 1.0 })), { 1 });
    ensureActivations(nn(Matrix(4, 1, { 0.0, 0.0, 1.0, 0.0 })), { 2 });
    ensureActivations(nn(Matrix(4, 1, { 0.0, 0.0, 1.0, 1.0 })), { 3 });
    ensureActivations(nn(Matrix(4, 1, { 0.0, 1.0, 0.0, 0.0 })), { 4 });
    ensureActivations(nn(Matrix(4, 1, { 0.0, 1.0, 0.0, 1.0 })), { 5 });
    ensureActivations(nn(Matrix(4, 1, { 0.0, 1.0, 1.0, 0.0 })), { 6 });
    ensureActivations(nn(Matrix(4, 1, { 0.0, 1.0, 1.0, 1.0 })), { 7 });
    ensureActivations(nn(Matrix(4, 1, { 1.0, 0.0, 0.0, 0.0 })), { 8 });
    ensureActivations(nn(Matrix(4, 1, { 1.0, 0.0, 0.0, 1.0 })), { 9 });
  }
};

struct NNDecToBinTest: NNTest {
  using NNTest::NNTest;
  virtual void run() {
    NN<2> nn(std::array<size_t, 2>{ { 10, 4 } });
    std::vector<std::pair<Matrix, Matrix>> trainingData;
    for (size_t i = 0; i < 1000; i++) {
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 1.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 }),
            Matrix(4, 1, { 0.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 1.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 }),
            Matrix(4, 1, { 0.0, 0.0, 0.0, 1.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 0.0, 1.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 }),
            Matrix(4, 1, { 0.0, 0.0, 1.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 0.0, 0.0, 1.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 }),
            Matrix(4, 1, { 0.0, 0.0, 1.0, 1.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 1.0, 
                            0.0, 0.0, 0.0, 0.0, 0.0 }),
            Matrix(4, 1, { 0.0, 1.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            1.0, 0.0, 0.0, 0.0, 0.0 }),
            Matrix(4, 1, { 0.0, 1.0, 0.0, 1.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 1.0, 0.0, 0.0, 0.0 }),
            Matrix(4, 1, { 0.0, 1.0, 1.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 1.0, 0.0, 0.0 }),
            Matrix(4, 1, { 0.0, 1.0, 1.0, 1.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 1.0, 0.0 }),
            Matrix(4, 1, { 1.0, 0.0, 0.0, 0.0 })));
      trainingData.push_back(
          std::pair<Matrix, Matrix>(
            Matrix(10, 1, { 0.0, 0.0, 0.0, 0.0, 0.0, 
                            0.0, 0.0, 0.0, 0.0, 1.0 }),
            Matrix(4, 1, { 1.0, 0.0, 0.0, 1.0 })));
    }
    nn.setMiniBatchSize(10);
    for (size_t i = 0; i < 200; i++) {
      nn(trainingData);
    }
    ensureActivations(nn(Matrix(10, 1,  { 1.0, 0.0, 0.0, 0.0, 0.0, 
                                          0.0, 0.0, 0.0, 0.0, 0.0 })), { });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 1.0, 0.0, 0.0, 0.0, 
                                          0.0, 0.0, 0.0, 0.0, 0.0 })), { 3 });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 0.0, 1.0, 0.0, 0.0, 
                                          0.0, 0.0, 0.0, 0.0, 0.0 })), { 2 });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 1.0, 0.0, 
                                          0.0, 0.0, 0.0, 0.0, 0.0 })), { 2, 3 });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 1.0, 
                                          0.0, 0.0, 0.0, 0.0, 0.0 })), { 1 });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                                          1.0, 0.0, 0.0, 0.0, 0.0 })), { 1, 3 });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                                          0.0, 1.0, 0.0, 0.0, 0.0 })), { 1, 2 });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                                          0.0, 0.0, 1.0, 0.0, 0.0 })), { 1, 2, 3 });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                                          0.0, 0.0, 0.0, 1.0, 0.0 })), { 0 });
    ensureActivations(nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                                          0.0, 0.0, 0.0, 0.0, 1.0 })), { 0, 3 });
  }
};

struct NNTestSuite: TestSuite {
  NNTestSuite(const char* name) : TestSuite(name) {
    addTest(new NNDecToBinTest("NNDecToBin"));
    addTest(new NNBinToDecTest("NNBinToDec"));
  }
};

DeclareTest(NNTestSuite, NNTests)
