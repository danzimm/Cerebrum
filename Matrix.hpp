/*
 * Made by DanZimm on Sun Mar 26 04:11:37 CDT 2017
 */
#pragma once

#include <cmath>
#include <initializer_list>
#include <string>
#include <utility>

#include <stdlib.h>
#include <string.h>

struct Matrix {
  enum GarbageTag {
    garbage
  };
  Matrix() : _data(nullptr), _rows(0), _columns(0) {}

  Matrix(size_t rows, size_t columns, double initialValue = 0.0) : _rows(rows), _columns(columns) {
    size_t size = columns * rows;
    if (initialValue != 0.0) {
      _data = reinterpret_cast<double *>(malloc(size * sizeof(double)));
      for (size_t i = 0; i < size; i++) {
        _data[i] = initialValue;
      }
    } else {
      _data = reinterpret_cast<double *>(calloc(size, sizeof(double)));
    }
  }

  Matrix(size_t rows, size_t columns, GarbageTag) : _rows(rows), _columns(columns) {
    size_t size = columns * rows;
    _data = reinterpret_cast<double *>(malloc(size * sizeof(double)));
  }

  template<typename RandGen, typename Dist>
  Matrix(size_t rows, size_t columns, RandGen gen, Dist dis) : _rows(rows), _columns(columns) {
    size_t size = columns * rows;
    _data = reinterpret_cast<double *>(malloc(size * sizeof(double)));
    for (size_t i = 0; i < size; i++) {
      _data[i] = dis(gen);
    }
  }

  Matrix(size_t rows, size_t columns, std::initializer_list<double> list) : _rows(rows), _columns(columns) {
    size_t size = rows * columns;
    if (list.size() != size) {
      std::string what = "Initializer list should be ";
      what += std::to_string(rows) + "x" + std::to_string(columns) + "=" + std::to_string(size);
      what += " long, but was " + std::to_string(list.size()) + " long";
      throw std::invalid_argument(what);
    }
    size_t bytes = size * sizeof(double);
    _data = reinterpret_cast<double *>(malloc(bytes));
    memcpy(_data, list.begin(), bytes);
  }

  ~Matrix() {
    free(_data);
  }

  Matrix(const Matrix& other) {
    size_t bytes;
    _rows = other._rows;
    _columns = other._columns;
    bytes = _rows * _columns * sizeof(double);
    _data = reinterpret_cast<double *>(malloc(bytes));
    memcpy(_data, other._data, bytes);
  }

  Matrix(Matrix&& other) {
    _rows = std::exchange(other._rows, 0);
    _columns = std::exchange(other._columns, 0);
    _data = std::exchange(other._data, nullptr);
  }
  
  Matrix& operator=(const Matrix& other) {
    size_t bytes;

    free(_data);
    _rows = other._rows;
    _columns = other._columns;
    bytes = _rows * _columns * sizeof(double);
    _data = reinterpret_cast<double *>(malloc(bytes));
    memcpy(_data, other._data, bytes);
    return *this;
  }

  Matrix& operator=(Matrix&& other) {
    _rows = std::exchange(other._rows, 0);
    _columns = std::exchange(other._columns, 0);
    free(_data);
    _data = std::exchange(other._data, nullptr);
    return *this;
  }

  friend Matrix operator*(double scalar, const Matrix& other) {
    return other * scalar;
  }

  Matrix operator*(const Matrix& other) const {
    if (_columns != other._rows) {
      std::string what("Cannot multiply matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns);
      what += " with matrix with dimensions ";
      what += std::to_string(other._rows) + "x" + std::to_string(other._columns);
      throw std::invalid_argument(what);
    }
    size_t columns = other._columns;
    size_t rows = _rows;
    Matrix result(rows, columns, garbage);
    for (size_t i = 0; i < rows; i++) {
      double* resultRow = result[i];
      const double* selfRow = (*this)[i];
      for (size_t j = 0; j < columns; j++) {
        double element = 0;
        for (size_t k = 0; k < _columns; k++) {
          element += selfRow[k] * other[k][j];
        }
        resultRow[j] = element;
      }
    }
    return result;
  }

  Matrix operator*(double scalar) const {
    Matrix result(_rows, _columns, garbage);
    for (size_t i = 0; i < _rows; i++) {
      const double* row = (*this)[i];
      double* resultRow = result[i];
      for (size_t j = 0; j < _columns; j++) {
        resultRow[j] = row[j] * scalar;
      }
    }
    return result;
  }

  Matrix& operator*=(const Matrix& other) {
    if (_columns != other._rows || _columns != other._columns) {
      std::string what("Cannot multiply matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns);
      what += " with matrix with dimensions ";
      what += std::to_string(other._rows) + "x" + std::to_string(other._columns);
      what += " in place";
      throw std::invalid_argument(what);
    }
    double tmp[_columns];
    for (size_t i = 0; i < _rows; i++) {
      double* selfRow = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        double element = 0;
        for (size_t k = 0; k < _columns; k++) {
          element += selfRow[k] * other[k][j];
        }
        tmp[j] = element;
      }
      memcpy(selfRow, tmp, sizeof(double) * _columns);
    }
    return *this;
  }

  Matrix& operator*=(double scalar) {
    for (size_t i = 0; i < _rows; i++) {
      double* row = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] *= scalar;
      }
    }
    return *this;
  }

  Matrix transpose() const {
    Matrix result(_columns, _rows, garbage);
    for (size_t i = 0; i < _rows; i++) {
      const double* row = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        result[j][i] = row[j];
      }
    }
    return result;
  }

  Matrix& transposeInPlace() {
    if (_rows != _columns) {
      std::string what("Cannot transpose matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns) + "in place";
      throw std::invalid_argument(what);
    }
    Matrix& self = *this;
    for (size_t i = 0; i < _rows; i++) {
      for (size_t j = _columns - 1; j > i; j--) {
        std::swap(self[i][j], self[j][i]);
      }
    }
    return self;
  }

  friend Matrix operator+(double scalar, const Matrix& other) {
    return other + scalar;
  }

  Matrix operator+(const Matrix& other) const {
    if (_rows != other._rows || _columns != other._columns) {
      std::string what("Cannot add matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns);
      what += " to matrix with dimensions ";
      what += std::to_string(other._rows) + "x" + std::to_string(other._columns);
      throw std::invalid_argument(what);
    }
    Matrix result(_rows, _columns, garbage);
    for (size_t i = 0; i < _rows; i++) {
      double* row = result[i];
      const double* thisRow = (*this)[i];
      const double* otherRow = other[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] = thisRow[j] + otherRow[j];
      }
    }
    return result;
  }
  
  Matrix operator+(double scalar) const {
    Matrix result(_rows, _columns, garbage);
    for (size_t i = 0; i < _rows; i++) {
      double* row = result[i];
      const double* thisRow = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] = thisRow[j] + scalar;
      }
    }
    return result;
  }

  Matrix& operator+=(const Matrix& other) {
    if (_rows != other._rows || _columns != other._columns) {
      std::string what("Cannot add matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns);
      what += " to matrix with dimensions ";
      what += std::to_string(other._rows) + "x" + std::to_string(other._columns);
      what += " in place";
      throw std::invalid_argument(what);
    }
    Matrix& self = *this;
    for (size_t i = 0; i < _rows; i++) {
      double* row = self[i];
      const double* otherRow = other[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] += otherRow[j];
      }
    }
    return self;
  }

  Matrix& operator+=(double scalar) {
    Matrix& self = *this;
    for (size_t i = 0; i < _rows; i++) {
      double* row = self[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] += scalar;
      }
    }
    return self;
  }

  friend Matrix operator-(double scalar, const Matrix& other) {
    return -(other - scalar);
  }

  Matrix operator-(const Matrix& other) const {
    if (_rows != other._rows || _columns != other._columns) {
      std::string what("Cannot subtract matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns);
      what += " to matrix with dimensions ";
      what += std::to_string(other._rows) + "x" + std::to_string(other._columns);
      throw std::invalid_argument(what);
    }
    Matrix result(_rows, _columns, garbage);
    for (size_t i = 0; i < _rows; i++) {
      double* row = result[i];
      const double* thisRow = (*this)[i];
      const double* otherRow = other[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] = thisRow[j] - otherRow[j];
      }
    }
    return result;
  }
  
  Matrix operator-(double scalar) const {
    Matrix result(_rows, _columns, garbage);
    for (size_t i = 0; i < _rows; i++) {
      double* row = result[i];
      const double* thisRow = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] = thisRow[j] - scalar;
      }
    }
    return result;
  }

  Matrix& operator-=(const Matrix& other) {
    if (_rows != other._rows || _columns != other._columns) {
      std::string what("Cannot subtract matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns);
      what += " to matrix with dimensions ";
      what += std::to_string(other._rows) + "x" + std::to_string(other._columns);
      what += " in place";
      throw std::invalid_argument(what);
    }
    Matrix& self = *this;
    for (size_t i = 0; i < _rows; i++) {
      double* row = self[i];
      const double* otherRow = other[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] -= otherRow[j];
      }
    }
    return self;
  }

  Matrix& operator-=(double scalar) {
    Matrix& self = *this;
    for (size_t i = 0; i < _rows; i++) {
      double* row = self[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] -= scalar;
      }
    }
    return self;
  }

  Matrix operator-() const {
    Matrix result(_rows, _columns, garbage);
    for (size_t i = 0; i < _rows; i++) {
      double* row = result[i];
      const double* thisRow = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] = -thisRow[j];
      }
    }
    return result;
  }

  Matrix& negateInPlace() {
    for (size_t i = 0; i < _rows; i++) {
      double* row = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] = -row[j];
      }
    }
    return *this;
  }

  Matrix elementWiseProduct(const Matrix& other) const {
    if (_rows != other._rows || _columns != other._columns) {
      std::string what("Cannot create element-wise product of matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns);
      what += " and matrix with dimensions ";
      what += std::to_string(other._rows) + "x" + std::to_string(other._columns);
      throw std::invalid_argument(what);
    }
    Matrix result(_rows, _columns, garbage);
    for (size_t i = 0; i < _rows; i++) {
      double* row = result[i];
      const double* thisRow = (*this)[i];
      const double* otherRow = other[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] = thisRow[j] * otherRow[j];
      }
    }
    return result;
  }

  Matrix& elementWiseProductInPlace(const Matrix& other) {
    if (_rows != other._rows || _columns != other._columns) {
      std::string what("Cannot create element-wise product of matrix with dimensions ");
      what += std::to_string(_rows) + "x" + std::to_string(_columns);
      what += " and matrix with dimensions ";
      what += std::to_string(other._rows) + "x" + std::to_string(other._columns);
      throw std::invalid_argument(what);
    }
    Matrix& self = *this;
    for (size_t i = 0; i < _rows; i++) {
      double* row = self[i];
      const double* otherRow = other[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] *= otherRow[j];
      }
    }
    return self;
  }

  template<typename Functor>
  Matrix apply(Functor f) const {
    Matrix result(*this);
    result.applyInPlace(f);
    return result;
  }

  template<typename Functor>
  Matrix& applyInPlace(Functor f) {
    for (size_t i = 0; i < _rows; i++) {
      double* row = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        row[j] = f(row[j]);
      }
    }
    return *this;
  }

  inline double* operator[](size_t row) {
    return &_data[row * _columns];
  }

  inline const double* operator[](size_t row) const {
    return &_data[row * _columns];
  }

  bool operator==(const Matrix& other) const {
    if (other._rows != _rows || other._columns != _columns) {
      return false;
    }
    return memcmp(_data, other._data, sizeof(double) * _columns * _rows) == 0;
  }

  bool operator!=(const Matrix& other) const {
    return !(*this == other);
  }

  bool equalsWithinBound(const Matrix& other, double bound) const {
    if (other._rows != _rows || other._columns != _columns) {
      return false;
    }
    for (size_t i = 0; i < _rows; i++) {
      const double* row = (*this)[i];
      const double* otherRow = other[i];
      for (size_t j = 0; j < _columns; j++) {
        if (std::abs(row[j] - otherRow[j]) >= bound) {
          return false;
        }
      }
    }
    return true;
  }

  inline size_t rows() const {
    return _rows;
  }
  
  inline size_t columns() const {
    return _columns;
  }
  
  std::string description() const {
    std::string result = "{";
    for (size_t i = 0; i < _rows; i++) {
      const double* row = (*this)[i];
      result += " { ";
      for (size_t j = 0; j < _columns; j++) {
        result += std::to_string(row[j]);
        if (j + 1 == _columns) {
          result += " ";
        } else {
          result += ", ";
        }
      }
      if (i + 1 != _rows) {
        result += "},";
      } else {
        result += "} ";
      }
    }
    result += "}";
    return result;
  }
 private:
  double *_data;
  size_t _rows;
  size_t _columns;
};

namespace std {
std::string to_string(const Matrix&);
}
