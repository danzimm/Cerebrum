/*
 * Made by DanZimm on Tue Mar 28 16:26:27 CDT 2017
 */

#include "IDXFile.hpp"
#include "NN.hpp"
#include "Test.hpp"

#include <set>

struct NNTest: Test {
  using Test::Test;
  void ensureActivations(const Matrix& m, const std::set<size_t>& activations, double bound=0.95) {
    auto end = activations.end();
#if 0
    std::cout << "Ensuring activations: " << m.description() << " - ";
    for (size_t index : activations) {
      std::cout << index << ", ";
    }
    std::cout << std::endl;
#endif
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
    NN<3> nn(std::array<size_t, 3>{ { 4, 8, 10 } });
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
    nn.setLearningRate(0.1);
    nn.setMiniBatchSize(10);
    for (size_t i = 0; i < 110; i++) {
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
    nn.setLearningRate(0.1);
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

struct NNDigitRecTest: NNTest {
  using NNTest::NNTest;
  virtual void run() {
    NN<3> nn(std::array<size_t, 3>{ { 784, 100, 10 } });

    Optional<IDXFile<uint8_t, 1>> optLabels = 
        IDXFile<uint8_t, 1>::fromFile("train-labels-idx1-ubyte.idx");
    Ensure(optLabels.value, "Load trainingLabels");
    const IDXFile<uint8_t, 1>& labels = *optLabels.value;

    Optional<IDXFile<uint8_t, 3>> optImgs = 
        IDXFile<uint8_t, 3>::fromFile("train-images-idx3-ubyte.idx");
    Ensure(optImgs.value, "Load trainingImages");
    const IDXFile<uint8_t, 3>& imgs = *optImgs.value;
    int numberTrainImgs = imgs.dimensionSize(0);
    int trainHeight = imgs.dimensionSize(1);
    int trainWidth = imgs.dimensionSize(2);
    EnsureEqual(numberTrainImgs,
                labels.dimensionSize(0),
                "Training labels contains same count as training images");

    Optional<IDXFile<uint8_t, 1>> optTstLabels = 
        IDXFile<uint8_t, 1>::fromFile("train-labels-idx1-ubyte.idx");
    Ensure(optTstLabels.value, "Load testLabels");
    const IDXFile<uint8_t, 1>& tstLabels = *optTstLabels.value;

    Optional<IDXFile<uint8_t, 3>> optTstImgs = 
        IDXFile<uint8_t, 3>::fromFile("train-images-idx3-ubyte.idx");
    Ensure(optTstImgs.value, "Load testImages");
    const IDXFile<uint8_t, 3>& tstImgs = *optTstImgs.value;
    int numberTstImgs = tstImgs.dimensionSize(0);
    //int tstHeight = tstImgs.dimensionSize(1);
    //int tstWidth = tstImgs.dimensionSize(2);
    EnsureEqual(numberTstImgs,
                tstLabels.dimensionSize(0),
                "Training labels contains same count as training images");

    std::vector<std::pair<Matrix, Matrix>> trainingData;
    for (int i = 0; i < numberTrainImgs; i++) {
      Matrix input(trainHeight * trainWidth, 1);
      for (int j = 0; j < trainHeight; j++) {
        for (int k = 0; k < trainWidth; k++) {
          input[j * trainWidth + k][0] = (double)imgs[{{i, j, k}}] / 255.0;
        }
      }
      Matrix output(10, 1);
      output[labels[{{ i }}]][0] = 1.0;
      trainingData.push_back({ std::move(input), std::move(output) });
    }
    nn.setMiniBatchSize(10);
    std::cout << "Training..." << std::endl;
    for (size_t i = 0; i < 30; i++) {
      nn(trainingData);
      std::cout << "Epoch " << i << std::endl;
    }
  }
};

struct NNTestSuite: TestSuite {
  NNTestSuite(const char* name) : TestSuite(name) {
    addTest(new NNDecToBinTest("NNDecToBin"));
    addTest(new NNBinToDecTest("NNBinToDec"));
    //addTest(new NNDigitRecTest("NNDigitRec"));
  }
};

DeclareTest(NNTestSuite, NNTests)
