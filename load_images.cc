#include "IDXFile.hpp"

#include <cassert>
#include <iostream>
#include <sys/ioctl.h>

static char characterMap[] = {
  ' ', // 0
  '.', // 1
  '^', // 2
  ':', // 3
  ';', // 4
  '*', // 5
  'O', // 6
  '@', // 7
};

static void usage(const char *prog) {
  std::cout << "Usage: " << prog << " [file_with_labels]" << std::endl;
}

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    usage(argv[0]);
    exit(1);
  }
  winsize w;
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) != 0) {
    perror("failed to get window size, assuming 0");
    w = { 0, 0, 0, 0 };
  }
  Optional<IDXFile<uint8_t, 3>> file = IDXFile<uint8_t, 3>::fromFile(argv[1]);
  if (!file.value) {
    usage(argv[0]);
    exit(1);
  }
  const IDXFile<uint8_t, 3>& images = *file.value;
  int numberImages = images.dimensionSize(0);
  int height = images.dimensionSize(1);
  int width = images.dimensionSize(2);
  int picsPerRow = w.ws_col == 0 ? 1 : w.ws_col / width;
  std::cout << "Found " << numberImages << " images with dimensions: " << width << " x " << height << std::endl;
  int ii = 0;
  for (int i = 0; i < numberImages; i += picsPerRow) {
    ii = i;
    for (int j = 0; j < height; j++) {
      for (int p = 0; p < picsPerRow; p++) {
        if (i + p >= numberImages) {
          continue;
        }
        for (int k = 0; k < width; k++) {
          uint8_t index = images[{{i + p, j, k}}];
          int characterMapIndex = 0;
          while (index >>= 1) characterMapIndex++;
          assert(characterMapIndex <= 7 && characterMapIndex >= 0);
          std::cout << characterMap[characterMapIndex];
        }
      }
      std::cout << std::endl;
    }
  }
  return 0;
}

