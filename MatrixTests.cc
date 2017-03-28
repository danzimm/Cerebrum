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
    Matrix should(2, 3, { 6.8, 8.1, 1.9,
                          12.32, 17.64, 4.66 });
    EnsureEqual(left * right, should, "Correct multiplication");

    left *= Matrix(2, 2,  { 2.0, 1.0,
                            1.2, 2.1 });  
    should = Matrix(2, 2, { 4.4, 5.2,
                            9.56, 8.23 });
    EnsureEqual(left, should, "Correct multiplication in place");

    should = Matrix(2, 2, { 4.84, 5.72,
                            10.516, 9.053 });
    EnsureEqual(1.1 * left, should, "Correct left scalar multiplication");
    EnsureEqual(left * 1.1, should, "Correct right scalar multiplication");

    left *= 1.1;
    EnsureEqual(left, should, "Correct inplace scalar multiplication");
  }
};

struct MatrixTransposeTest: Test {
  using Test::Test;
  virtual void run() {
    Matrix left(4, 2, { 1.0, 2.0,
                        3.0, 4.0,
                        5.0, 6.0,
                        7.0, 8.0 });
    Matrix should(2, 4, { 1.0, 3.0, 5.0, 7.0,
                          2.0, 4.0, 6.0, 8.0 });
    EnsureEqual(left.transpose(), should, "Correct transpose");

    left = Matrix(4, 4, { 1.0, 2.0, 3.0, 4.0,
                          5.0, 6.0, 7.0, 8.0,
                          9.0, 8.8, 7.7, 6.6,
                          5.5, 4.4, 3.3, 2.2 });
    should = Matrix(4, 4, { 1.0, 5.0, 9.0, 5.5,
                            2.0, 6.0, 8.8, 4.4,
                            3.0, 7.0, 7.7, 3.3,
                            4.0, 8.0, 6.6, 2.2 });
    left.transposeInPlace();
    EnsureEqual(left, should, "Correct transpose in place");
  }
};

struct MatrixNegationTest: Test {
  using Test::Test;
  virtual void run() {
    Matrix matrix(3, 2, { 2.2, -1.3, 
                          -2.1, 1.1,
                          1.9, -1.8 });
    Matrix should(3, 2, { -2.2, 1.3,
                          2.1, -1.1,
                          -1.9, 1.8 });
    EnsureEqual(-matrix, should, "Negation should properly propogate"); 
    matrix.negateInPlace();
    EnsureEqual(matrix, should, "Correct negation in place");
  }
};

struct MatrixAdditionTest: Test {
  using Test::Test;
  virtual void run() {
    Matrix left(3, 2, { 2.2, 1.3, 
                        2.1, 1.1,
                        1.9, 1.8 });
    Matrix right(3, 2,  { 2.1, 0.7,
                          0.8, 0.8,
                          1.1, 1.2 });
    Matrix should(3, 2, { 4.3, 2.0,
                          2.9, 1.9,
                          3.0, 3.0 });
    EnsureEqual(left + right, should, "Correct addition");
    left += right;
    EnsureEqual(left, should, "Correct addition in place");

    should = Matrix(3, 2, { 5.4, 3.1,
                            4.0, 3.0,
                            4.1, 4.1 });
    EnsureEqual(left + 1.1, should, "Correct addition right scalar");
    EnsureEqual(1.1 + left, should, "Correct addition left scalar");
    left += 1.1;
    EnsureEqual(left, should, "Correct addition in place scalar");
  }
};

struct MatrixSubtractionTest: Test {
  using Test::Test;
  virtual void run() {
    Matrix left(3, 2, { 2.2, 1.3, 
                        2.1, 1.1,
                        1.9, 1.8 });
    Matrix right(3, 2,  { 2.1, 1.7,
                          0.8, 1.8,
                          1.1, 2.2 });
    Matrix should(3, 2, { 0.1, -0.4,
                          1.3, -0.7,
                          0.8, -0.4 });
    EnsureEqual(left - right, should, "Correct subtraction");
    left -= right;
    EnsureEqual(left, should, "Correct subtraction in place");

    should = Matrix(3, 2, { -1.0, -1.5,
                            0.2, -1.8,
                            -0.3, -1.5 });
    EnsureEqual(left - 1.1, should, "Correct subtraction scalar right");
    EnsureEqual(1.1 - left, -should, "Correct subtraction scalar left");
    left -= 1.1;
    EnsureEqual(left, should, "Correct subtraction scalar in place");
  }
};

struct MatrixTestSuite: TestSuite {
  MatrixTestSuite(const char* name) : TestSuite(name) {
    addTest(new MatrixMultiplicationTest("matMult"));
    addTest(new MatrixTransposeTest("matTrans"));
    addTest(new MatrixNegationTest("matNeg"));
    addTest(new MatrixAdditionTest("matAdd"));
    addTest(new MatrixSubtractionTest("matSub"));
  }
};

DeclareTest(MatrixTestSuite, MatrixTests)
