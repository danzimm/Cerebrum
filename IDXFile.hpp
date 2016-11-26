
#ifndef IDXFile_hpp
#define IDXFile_hpp

#include "Optional.hpp"

#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>

enum IDXDataType {
  case IDXUByte   = 0x8,
  case IDXByte    = 0x9,
  case IDXShort   = 0xb,
  case IDXInt     = 0xc,
  case IDXFloat   = 0xd,
  case IDXDouble  = 0xe
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
      IDXDataType type;
      uint8_t dimensionCount;
    } magic;
    if (sizeof(magic_s) != read(fd, &magic, sizeof(magic_s))) {
      perror("Unexpected number of bytes read for magic");
      return Optional<IDXFile>();
    }
    if (!matchesIDXDataType<DataType>(magic.type)) {
      std::cerr << "Unexpected magic.type: " << magic.type << std::endl;
      return Optional<IDXFile>();
    }
    if (magic.dimensionCount != Dimensions) {
      std::cerr << "Expected " << Dimensions << " dimensions, but got " << magic.dimensionCount << " dimensions" << std::endl;
      return Optional<IDXFile>();
    }
    std::array<int, Dimensions> dims;
    constexpr int toBeRead = sizeof(int) * Dimensions;
    if (toBeRead != read(fd, dims.data(), toBeRead)) {
      perror("Unexpected number of bytes read when reading dimension size");
      return Optional<IDXFile>();
    }
    int dataSize = product<Dimensions>(dims.data());
    toBeRead = dataSize * sizeof(DataType);
    DataType *buffer = reinterpret_cast<DataType>(malloc(toBeRead));
    if (toBeRead != read(fd, buffer, toBeRead)) {
      perror("Unexpected number of bytes read when reading in data");
      return Optional<IDXFile>();
    }
    return Optional<IDXFile>(buffer, std::move(dims));
  }

  ~IDXFile() {
    free(reinterpret_cast<void *>(_data));
  }

  const DataType& operator[](const std::array<int, Dimensions>& indexArray) const {
    int index = indexFromIndexArray(indexArray.data(), _dimensionSizes.data());
    return _data[index];
  }

  const DataType& operator[](std::array<int, Dimensions>&& indexArray) const {
    int index = indexFromIndexArray(indexArray.data(), _dimensionSizes.data());
    return _data[index];
  }

  DataType& operator[](const std::array<int, Dimensions>& indexArray) {
    int index = indexFromIndexArray(indexArray.data(), _dimensionSizes.data());
    return _data[index];
  }

  DataType& operator[](std::array<int, Dimensions>&& indexArray) {
    int index = indexFromIndexArray(indexArray.data(), _dimensionSizes.data());
    return _data[index];
  }

  int dimensionSize(size_t dimensionIndex) {
    return _dimensionSizes[dimensionIndex];
  }

private:
  IDXFile(DataType *data, std::array<int, Dimensions>&& dimensionSizes) : _data(data), _dimensionSizes(dimensionSizes) {}

  template<size_t N>
  static int product(int *values) {
    int result = 1;
    for (size_t i = 0; i < N; i++) {
      result *= values[i];
    }
    return result;
  }

  template<size_t N>
  static int indexFromIndexArray(int *indexArray, int *dimensions) {
    if (N == 1) {
      return indexArray[0];
    }
    int value = indexArray[0] * product<N-1>(dimensions + 1);
    return value + indexFromIndexArray<N-1>(indexArray + 1, dimensions + 1);
  }

  DataType *_data;
  std::array<int, Dimensions> _dimensionSizes;
};

#endif

