
#ifndef IDXFile_hpp
#define IDXFile_hpp

#include "Optional.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace _detail {

enum {
  IDXUByte   = 0x8,
  IDXByte    = 0x9,
  IDXShort   = 0xb,
  IDXInt     = 0xc,
  IDXFloat   = 0xd,
  IDXDouble  = 0xe
};

typedef uint8_t IDXDataType;

template<typename T>
bool matchesIDXDataType(IDXDataType type) {
  return false;
}

template<>
bool matchesIDXDataType<uint8_t>(IDXDataType type) {
  return type == IDXUByte;
}

template<>
bool matchesIDXDataType<int8_t>(IDXDataType type) {
  return type == IDXByte;
}

template<>
bool matchesIDXDataType<int16_t>(IDXDataType type) {
  return type == IDXShort;
}

template<>
bool matchesIDXDataType<int32_t>(IDXDataType type) {
  return type == IDXInt;
}

template<>
bool matchesIDXDataType<float>(IDXDataType type) {
  return type == IDXFloat;
}

template<>
bool matchesIDXDataType<double>(IDXDataType type) {
  return type == IDXDouble;
}


template<size_t N>
int product(const int *values) {
  int result = 1;
  for (size_t i = 0; i < N; i++) {
    result *= values[i];
  }
  return result;
}

template<size_t N>
int indexFromIndexArray(const int *indexArray, const int *dimensions) {
  int value = indexArray[0] * product<N-1>(dimensions + 1);
  return value + indexFromIndexArray<N-1>(indexArray + 1, dimensions + 1);
}

template<>
int indexFromIndexArray<1>(const int *indexArray, const int *dimensions) {
  return indexArray[0];
}

int32_t byteSwap(int32_t inval) {
  return (((inval & 0xff000000) >> 24) |
          ((inval & 0x00ff0000) >> 8)  |
          ((inval & 0x0000ff00) << 8)  |
          ((inval & 0x000000ff) << 24));
}

bool bounded(int left, int right) {
  assert(left >= 0 && left < right);
  return left >= 0 && left < right;
}

}

template<typename DataType, size_t Dimensions>
struct IDXFile {
  static Optional<IDXFile> fromFile(const char *file) {
    struct stat s;
    if (stat(file, &s) != 0) {
      return Optional<IDXFile>();
    }
    int fd = open(file, O_RDONLY);
    if (fd < 0) {
      perror("Failed to open IDXFile");
      return Optional<IDXFile>();
    }
    struct magic_s {
      uint16_t zero;
      _detail::IDXDataType type;
      uint8_t dimensionCount;
    } magic;
    if (sizeof(magic_s) != read(fd, &magic, sizeof(magic_s))) {
      perror("Unexpected number of bytes read for magic");
      return Optional<IDXFile>();
    }
    if (magic.zero != 0) {
      std::cerr << "Expected proper zero'd bytes in magic: " << (void *)(uintptr_t)magic.zero << std::endl;
    }
    if (!_detail::matchesIDXDataType<DataType>(magic.type)) {
      std::cerr << "Unexpected magic.type: " << magic.type << std::endl;
      return Optional<IDXFile>();
    }
    if (magic.dimensionCount != Dimensions) {
      std::cerr << "Expected " << Dimensions << " dimensions, but got " << (size_t)magic.dimensionCount << " dimensions" << std::endl;
      return Optional<IDXFile>();
    }
    std::array<int, Dimensions> dims;
    int red = read(fd, dims.data(), sizeof(int) * Dimensions);
    if (sizeof(int) * Dimensions != red) {
      std::cout << "Expected to read " << sizeof(int) * Dimensions << " bytes but read in " << red << " bytes when reading dimensions" << std::endl;
      return Optional<IDXFile>();
    }
    for (size_t i = 0; i < Dimensions; i++) {
      dims[i] = _detail::byteSwap(dims[i]);
    }
    int dataSize = _detail::product<Dimensions>(dims.data());
    int toBeRead = dataSize * sizeof(DataType);
    DataType *buffer = reinterpret_cast<DataType *>(malloc(toBeRead));
    red = read(fd, buffer, toBeRead);
    if (toBeRead != red) {
      std::cout << "Expected to read " << toBeRead << " bytes but read in " << red << " bytes when reading data" << std::endl;
      return Optional<IDXFile>();
    }
    return Optional<IDXFile>(IDXFile(buffer, std::move(dims), toBeRead));
  }

  IDXFile(IDXFile&& other) : _data(NULL), _dimensionSizes(other._dimensionSizes) {
    std::swap(other._data, _data);
  }

  IDXFile(const IDXFile& other) : _data(NULL), _dimensionSizes(other._dimensionSizes) {
    int dataSize = _detail::product<Dimensions>(_dimensionSizes.data());
    int toBeCopied = dataSize * sizeof(DataType);
    _data = reinterpret_cast<DataType *>(malloc(toBeCopied));
    memcpy(_data, other._data, toBeCopied);
  }

  ~IDXFile() {
    free(reinterpret_cast<void *>(_data));
  }

  const DataType& operator[](const std::array<int, Dimensions>& indexArray) const {
    _checkIndexInBounds(indexArray, std::make_index_sequence<Dimensions>{});
    int index = _detail::indexFromIndexArray<Dimensions>(indexArray.data(), _dimensionSizes.data());
    return _data[index];
  }

  const DataType& operator[](std::array<int, Dimensions>&& indexArray) const {
    _checkIndexInBounds(indexArray, std::make_index_sequence<Dimensions>{});
    int index = _detail::indexFromIndexArray<Dimensions>(indexArray.data(), _dimensionSizes.data());
    return _data[index];
  }

  DataType& operator[](const std::array<int, Dimensions>& indexArray) {
    _checkIndexInBounds(indexArray, std::make_index_sequence<Dimensions>{});
    int index = _detail::indexFromIndexArray<Dimensions>(indexArray.data(), _dimensionSizes.data());
    return _data[index];
  }

  DataType& operator[](std::array<int, Dimensions>&& indexArray) {
    _checkIndexInBounds(indexArray, std::make_index_sequence<Dimensions>{});
    int index = _detail::indexFromIndexArray<Dimensions>(indexArray.data(), _dimensionSizes.data());
    return _data[index];
  }

  int dimensionSize(size_t dimensionIndex) const {
    return _dimensionSizes[dimensionIndex];
  }

private:
  IDXFile(DataType *data, std::array<int, Dimensions>&& dimensionSizes, size_t count) : _data(data), _dimensionSizes(dimensionSizes) {}

  template<size_t ...Is>
  void _checkIndexInBounds(const std::array<int, Dimensions>& indexArray, std::index_sequence<Is...>) const {
    std::array<bool, Dimensions> inBounds = {{ _detail::bounded(indexArray[Is], _dimensionSizes[Is])... }};
    bool result = std::accumulate(inBounds.begin(), inBounds.end(), true, [](const bool& left, const bool& right) { return left && right; });
    if (!result) {
      throw std::out_of_range("Attempted to access element out of bounds");
    }
  }

  DataType *_data;
  std::array<int, Dimensions> _dimensionSizes;
};

#endif

