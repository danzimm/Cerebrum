/*
 * Made by DanZimm on Sun Mar 26 18:19:27 CDT 2017
 */
#include "Test.hpp"

#include <cmath>

template<>
void Test::ensureEqual(const double& left, const double& right, std::string message) {
  if (std::abs(left - right) > 0.0000001) {
    std::string info = message + " - " + std::to_string(left) + " != " + std::to_string(right);
    _didFail(info);
  }
}

int main(int argc, const char* const argv[]) {
  return TestSuite::defaultSuite()._run() ? 0 : 1;
}

