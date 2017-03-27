/*
 * Made by DanZimm on Sun Mar 26 04:11:37 CDT 2017
 */
#pragma once

#include <cassert>
#include <cmath>
#include <initializer_list>
#include <string>
#include <utility>

#include <stdlib.h>
#include <string.h>

struct Matrix {
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

  Matrix(size_t rows, size_t columns, std::initializer_list<double> list) : _rows(rows), _columns(columns) {
    size_t size = rows * columns;
    assert(list.size() == size);
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
  
  Matrix& operator=(Matrix& other) {
    size_t bytes;

    free(_data);
    _rows = std::exchange(other._rows, 0);
    _columns = std::exchange(other._columns, 0);
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

  Matrix operator*(Matrix& other) const {
    assert(_columns == other._rows);
    size_t columns = other._columns;
    size_t rows = _rows;
    Matrix result(rows, columns);
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

  Matrix& operator*=(Matrix& other) {
    assert(_columns == other._rows);
    assert(_columns == other._columns);
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

  Matrix transpose() const {
    Matrix result(_columns, _rows);
    for (size_t i = 0; i < _rows; i++) {
      const double* row = (*this)[i];
      for (size_t j = 0; j < _columns; j++) {
        result[j][i] = row[j];
      }
    }
    return result;
  }

  double* operator[](size_t row) {
    return &_data[row * _columns];
  }

  const double* operator[](size_t row) const {
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
  
  size_t rows() const {
    return _rows;
  }
  
  size_t columns() const {
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
          result += ", ";
        } else {
          result += " ";
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
