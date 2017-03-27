/*
 * Made by DanZimm on Sun Mar 26 19:06:43 CDT 2017
 */

#include "Matrix.hpp"
#include "Test.hpp"

template<>
void Test::ensureEqual(const Matrix& left, const Matrix& right, std::string message) {
  if (!left.equalsWithinBound(right, 0.0000001)) {
    std::string info = message + " - " + std::to_string(left) + " != " + std::to_string(right);
    _didFail(info);
  }
}

struct MatrixMultiplicationTest: Test {
  using Test::Test;
  virtual void run() {
    Matrix left(2, 2, { 1.0, 2.0, 
                        3.4, 2.3 });
    Matrix right(2, 3,  { 2.0, 3.7, 1.1, 
                          2.4, 2.2, 0.4 });
    Matrix result = left * right;
    Matrix should(2, 3, { 6.8, 8.1, 1.9,
                          12.32, 17.64, 4.66 });
    EnsureEqual(result, should, "Correct Multiplication");
  }
};

struct MatrixMultiplicationInplaceTest: Test {
  using Test::Test;
  virtual void run() {
    Matrix left(2, 3, { 1.0, 2.0, 3.0, 
                        3.0, 2.0, 1.0 });
    Matrix right(3, 3,  { 0.2, 0.4, 1.2, 
                          3.1, 1.9, 1.8,
                          4.8, 3.2, 20.0 });
    left *= right;
    Matrix should(2, 3, { 20.8, 13.8, 64.8,
                          11.6, 8.2, 27.2 });
    EnsureEqual(left, should, "Correct in place multi");
  }
};

struct MatrixTranspose: Test {
  using Test::Test;
  virtual void run() {
    Matrix left(4, 2, { 1.0, 2.0,
                        3.0, 4.0,
                        5.0, 6.0,
                        7.0, 8.0 });
    Matrix trans = left.transpose();
    Matrix should(2, 4, { 1.0, 3.0, 5.0, 7.0,
                          2.0, 4.0, 6.0, 8.0 });
    EnsureEqual(trans, should, "Correct transpose");
  }
};

struct MatrixTestSuite: TestSuite {
  MatrixTestSuite(const char* name) : TestSuite(name) {
    addTest(new MatrixMultiplicationTest("matMult"));
    addTest(new MatrixMultiplicationInplaceTest("matInplaceMult"));
    addTest(new MatrixTranspose("matTranspose"));
    /*
    addTest(new MatrixNegateTest("matNeg"));
    addTest(new MatrixAdditionTest("matAdd"));
    addTest(new MatrixSubtractionTest("matSub"));
    addTest(new MatrixTransposeTest("matTrans"));
    */
  }
};

DeclareTest(MatrixTestSuite, MatrixTests)
