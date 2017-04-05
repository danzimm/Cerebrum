/*
 * Made by DanZimm on Tue Mar 28 16:26:27 CDT 2017
 */

#include "IDXFile.hpp"
#include "NN.hpp"
#include "Test.hpp"

#include <set>

using ActivationSet = const std::set<size_t>;

template<typename Activator, typename Cost>
struct Bounds {
  static inline std::pair<double, double> bounds(size_t index,
                                                 size_t totalOut,
                                                 ActivationSet& activations) {
    return activations.find(index) != activations.end() 
          ? std::pair<double, double>{ 1.0, 0.95 } 
          : std::pair<double, double>{ 0.05, 0.0 };
  }
};

template<typename Cost>
struct Bounds<SoftMax, Cost> {
  static inline std::pair<double, double> bounds(size_t index,
                                                 size_t totalOut,
                                                 ActivationSet& activations) {
    size_t numberAct = activations.size();
    if (numberAct == 0) {
      return std::pair<double, double>{ 0.40, 0.0 };
    }
    if (activations.find(index) != activations.end()) {
      double upper = 1.0 / (double)numberAct - 0.1;
      return std::pair<double, double>{ 1.0, upper };
    } else {
      return std::pair<double, double>{ 0.05, 0.0 };
    }
  }
};

struct NNTest: Test {
  using Test::Test;
  template<typename Activator, typename Cost>
  void ensureActivations(const Matrix& m, ActivationSet& activations) {
    size_t height = m.rows();
#if 0
    std::cout << "Ensuring ";
    for (auto& act : activations) {
      std::cout << act << " ";
    }
    std::cout << m.description() << std::endl;
#endif
    for (size_t i = 0; i < height; i++) {
      double value = m[i][0];
      Ensure(isfinite(value), "Finite");
      auto bounds = Bounds<Activator, Cost>::bounds(i, height, activations);
      EnsureGreaterThanEqual(value, bounds.second, "Activation when expected");
      EnsureLessThanEqual(value, bounds.first, "No activation when expected");
    }
  }

  void ensureMaxActivation(const Matrix& m, size_t index) {
    size_t maxIndex = maxActivation(m).first;
    EnsureEqual(maxIndex, index, "Max index occurs where expected");
  }

  std::pair<size_t, double> maxActivation(const Matrix& m) {
    size_t maxIndex = (size_t)-1;
    double max = 0.0;
    for (size_t i = 0; i < m.rows(); i++) {
      double tmp = m[i][0];
      if (tmp > max) {
        max = tmp;
        maxIndex = i;
      }
    }
    return { maxIndex, max };
  }
};

template<typename Activator, typename Cost, size_t numberEpochs=2, size_t learningRateNum=3, size_t learningRateDenom=1, size_t miniBatchSize=10>
struct NNDecToBinTest: NNTest {
  using NNTest::NNTest;
  virtual void run() {
    NN<2, Cost, Activator> nn(std::array<size_t, 2>{ { 10, 4 } });
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
    nn.setLearningRate((double)learningRateNum / (double)learningRateDenom);
    nn.setMiniBatchSize(miniBatchSize);
    for (size_t i = 0; i < numberEpochs; i++) {
      nn(trainingData);
    }
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 1.0, 0.0, 0.0, 0.0, 0.0, 
                             0.0, 0.0, 0.0, 0.0, 0.0 })), { });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 1.0, 0.0, 0.0, 0.0, 
                             0.0, 0.0, 0.0, 0.0, 0.0 })), { 3 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 0.0, 1.0, 0.0, 0.0, 
                             0.0, 0.0, 0.0, 0.0, 0.0 })), { 2 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 1.0, 0.0, 
                             0.0, 0.0, 0.0, 0.0, 0.0 })), { 2, 3 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 1.0, 
                             0.0, 0.0, 0.0, 0.0, 0.0 })), { 1 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                             1.0, 0.0, 0.0, 0.0, 0.0 })), { 1, 3 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                             0.0, 1.0, 0.0, 0.0, 0.0 })), { 1, 2 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                             0.0, 0.0, 1.0, 0.0, 0.0 })), { 1, 2, 3 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                             0.0, 0.0, 0.0, 1.0, 0.0 })), { 0 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(10, 1,  { 0.0, 0.0, 0.0, 0.0, 0.0, 
                             0.0, 0.0, 0.0, 0.0, 1.0 })), { 0, 3 });
  }
};

template<typename Activator, typename Cost, size_t numberEpochs=2, size_t learningRateNum=25, size_t learningRateDenom=1, size_t miniBatchSize=10>
struct NNBinToDecTest: NNTest {
  using NNTest::NNTest;
  virtual void run() {
    NN<2, Cost, Activator> nn(std::array<size_t, 2>{ { 4, 10 } });
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
    nn.setLearningRate((double)learningRateNum / (double)learningRateDenom);
    nn.setMiniBatchSize(miniBatchSize);
    for (size_t i = 0; i < numberEpochs; i++) {
      nn(trainingData);
    }
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 0.0, 0.0, 0.0, 0.0 })), { 0 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 0.0, 0.0, 0.0, 1.0 })), { 1 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 0.0, 0.0, 1.0, 0.0 })), { 2 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 0.0, 0.0, 1.0, 1.0 })), { 3 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 0.0, 1.0, 0.0, 0.0 })), { 4 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 0.0, 1.0, 0.0, 1.0 })), { 5 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 0.0, 1.0, 1.0, 0.0 })), { 6 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 0.0, 1.0, 1.0, 1.0 })), { 7 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 1.0, 0.0, 0.0, 0.0 })), { 8 });
    ensureActivations<Activator, Cost>
        (nn(Matrix(4, 1, { 1.0, 0.0, 0.0, 1.0 })), { 9 });
  }
};

struct NNDigitRecTest: NNTest {
  using NNTest::NNTest;
  virtual void run() {}
  virtual void run(size_t indents) {
    NN<3> nn(std::array<size_t, 3>{ { 784, 100, 10 } },
                      std::normal_distribution<>());
    nn.setLearningRate(3.0);
    nn.setMiniBatchSize(10);

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
        IDXFile<uint8_t, 1>::fromFile("t10k-labels-idx1-ubyte.idx");
    Ensure(optTstLabels.value, "Load testLabels");
    const IDXFile<uint8_t, 1>& tstLabels = *optTstLabels.value;

    Optional<IDXFile<uint8_t, 3>> optTstImgs = 
        IDXFile<uint8_t, 3>::fromFile("t10k-images-idx3-ubyte.idx");
    Ensure(optTstImgs.value, "Load testImages");
    const IDXFile<uint8_t, 3>& tstImgs = *optTstImgs.value;
    int numberTstImgs = tstImgs.dimensionSize(0);
    EnsureEqual(numberTstImgs,
                tstLabels.dimensionSize(0),
                "Training labels contains same count as training images");

    std::vector<std::pair<Matrix, Matrix>> trainingData;
    for (int i = 0; i < numberTrainImgs; i++) {
      Matrix input(trainHeight * trainWidth, 1, Matrix::garbage);
      for (int j = 0; j < trainHeight; j++) {
        for (int k = 0; k < trainWidth; k++) {
          input[j * trainWidth + k][0] = (double)imgs[{{i, j, k}}] / 255.0;
        }
      }
      Matrix output(10, 1);
      output[labels[{{ i }}]][0] = 1.0;
      trainingData.push_back({ std::move(input), std::move(output) });
    }
    std::vector<Matrix> tstData;
    for (int i = 0; i < numberTstImgs; i++) {
      Matrix input(trainHeight * trainWidth, 1, Matrix::garbage);
      for (int j = 0; j < trainHeight; j++) {
        for (int k = 0; k < trainWidth; k++) {
          input[j * trainWidth + k][0] = (double)tstImgs[{{i, j, k}}] / 255.0;
        }
      }
      tstData.push_back(std::move(input));
    }
    std::string prefix;
    for (size_t i = 0; i < indents; i++) {
      prefix += "  ";
    }
    size_t nepoch = 30;
    int numberCorrect = 0;
    for (size_t i = 0; i < nepoch; i++) {
      nn(trainingData);
      std::cout << prefix << "  Epoch " << i << ": ";
      numberCorrect = 0;
      for (int i = 0; i < numberTstImgs; i++) {
        Matrix m = nn(tstData[i]);
        auto result = maxActivation(m);
        if ((uint8_t)result.first == tstLabels[{{ i }}]) {
          numberCorrect += 1;
        }
      }
      std::cout << numberCorrect << " / " << numberTstImgs << std::endl;
    }
    int ninetyFivePercent = (numberTstImgs * 95) / 100;
    EnsureGreaterThan(numberCorrect, ninetyFivePercent, "At least 95% imgs passed");
  }
};

struct NNTestSuite: TestSuite {
  NNTestSuite(const char* name) : TestSuite(name) {
    addTest(new NNDecToBinTest<Sigmoid, MSE>("NNDecToBin-Sigmoid-MSE"));
    addTest(new NNBinToDecTest<Sigmoid, MSE>("NNBinToDec-Sigmoid-MSE"));
    addTest(new NNDecToBinTest<Sigmoid, CrossEntropy>("NNDecToBin-Sigmoid-CrossEntropy"));
    addTest(new NNBinToDecTest<Sigmoid, CrossEntropy>("NNBinToDec-Sigmoid-CrossEntropy"));
    addTest(new NNDecToBinTest<SoftMax, MSE, 1>("NNDecToBin-SoftMax-MSE"));
    addTest(new NNBinToDecTest<SoftMax, MSE, 1>("NNBinToDec-SoftMax-MSE"));
    addTest(new NNDigitRecTest("NNDigitRec"));
  }
};

DeclareTest(NNTestSuite, NNTests)
