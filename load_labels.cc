/*
 * Made by DanZimm on Sun Mar 26 03:51:23 CDT 2017
 */
#include "IDXFile.hpp"

#include <iostream>

static void usage(const char *prog) {
  std::cout << "Usage: " << prog << " [file_with_labels]" << std::endl;
}

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    usage(argv[0]);
    exit(1);
  }
  Optional<IDXFile<uint8_t, 1>> file = IDXFile<uint8_t, 1>::fromFile(argv[1]);
  if (!file.value) {
    usage(argv[0]);
    exit(1);
  }
  const IDXFile<uint8_t, 1>& labels = *file.value;
  for (size_t i = 0; i < labels.dimensionSize(0); i++) {
    std::cout << (size_t)labels[{ { static_cast<int>(i) } }] << " ";
  }
  std::cout << std::endl;
  return 0;
}

