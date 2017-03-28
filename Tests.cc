/*
 * Made by DanZimm on Sun Mar 26 18:19:27 CDT 2017
 */
#include "Test.hpp"

#include <cmath>
#include <iostream>

#include <execinfo.h>

template<>
void Test::ensureEqual(const double& left, const double& right, std::string message) {
  if (std::abs(left - right) > 0.0000001) {
    std::string info = message + " - " + std::to_string(left) + " != " + std::to_string(right);
    _didFail(info);
  }
}

void term() {
  void* callstack[128];
  int frames = backtrace(callstack, sizeof(callstack)/sizeof(void*));
  char** strs = backtrace_symbols(callstack, frames);
  for (int i = 0; i < frames; ++i) {
    std::cout << strs[i] << std::endl;
  }
  free(strs);
  exit(1);
}

int main(int argc, const char* const argv[]) {
  std::set_terminate(term);
  return TestSuite::defaultSuite()._run() ? 0 : 1;
}

