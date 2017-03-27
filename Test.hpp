/*
 * Made by DanZimm on Sun Mar 26 17:32:36 CDT 2017
 */
#pragma once

#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include <mach/mach.h>
#include <mach/mach_time.h>
#include <unistd.h>

#define DeclareTest(cls, name) \
  cls __test ## name( # name ); \
  __attribute__((constructor)) \
  static void _init ## name () { \
    TestSuite::defaultSuite().addTest( &__test ## name ); \
  }

#define _fixUpMsg(func, msg, ...) \
  { std::string __fixtmp = std::string("'") + (msg) + "'"; __fixtmp += " at " __FILE__ ":"; __fixtmp += std::to_string(__LINE__); func (__VA_ARGS__, __fixtmp); }

#define EnsureEqual(a, b, msg) \
  _fixUpMsg(ensureEqual, #a " == " #b ": " msg, a, b)

#define Ensure(val, msg) \
  _fixUpMsg(ensure, msg, val)

#define EnsureNot(val, msg) \
  _fixUpMsg(ensureNot, msg, val)

struct Test {
  Test(const char* name) : _name(name), _failed(false) {}

  virtual void run(size_t indents) { run(); }
  virtual void run() = 0;

  void ensure(bool value, std::string message) {
    if (!value) {
      _didFail(message);
    }
  }

  void ensureNot(bool value, std::string message) {
    if (value) {
      _didFail(message);
    }
  }

  template<typename T, typename U>
  void ensureEqual(const T& left, const U& right, std::string message) {
    if (left != right) {
      std::string info = message + " - " + std::to_string(left) + " != " + std::to_string(right);
      _didFail(info);
    }
  }

  bool _run(size_t indents=0) {
    static mach_timebase_info_data_t timebaseInfo;
    uint64_t start;
    uint64_t end;
    uint64_t elapsed;
    double secElapsed;
    
    for (size_t i = 0; i < indents; i++) {
      std::cout << "  ";
    }
    std::cout << "Running " << _name << ": " << std::endl;
    start = mach_absolute_time();
    try {
      run(indents);
    } catch(std::runtime_error& error) {
    }
    end = mach_absolute_time();
    bool success = !std::exchange(_failed, false);
    if (timebaseInfo.denom == 0) {
      mach_timebase_info(&timebaseInfo);
    }
    elapsed = end - start;
    secElapsed = (double)(elapsed * timebaseInfo.numer) / (double)(1000000000 * timebaseInfo.denom);

    for (size_t i = 0; i < indents + 1; i++) {
      std::cout << "  ";
    }
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\x1b[0;" << (success ? "32m" : "31m");
    std::cout << (success ? "Success in " : "Failed in ") << secElapsed << " seconds";
    if (!success) {
      std::cout << ": " << _failedMessage;
      _failedMessage.clear();
    }
    std::cout << "\x1b[0m" << std::defaultfloat << std::endl;
    return success;
  }
 protected:
  const char* _name;
  std::string _failedMessage;
  bool _failed;
 private:
  void _didFail(std::string& message) {
    _failed = true;
    _failedMessage = message;
    throw std::runtime_error("failed tests");
  }
};

struct TestSuite : Test {
  enum DefaultToken { tag };

  static TestSuite& defaultSuite() {
    static TestSuite suite = TestSuite(tag);
    return suite;
  }
  static void runDefaultSuite() {
    defaultSuite()._run();
  }

  TestSuite(const char* name) : Test(name), _default(false) {}

  virtual void run(size_t indents) {
    for (auto test : _tests) {
      if (!test->_run(indents + 1)) {
        _failed = true;
        break;
      }
    }
  }
  virtual void run() { run(0); }

  void addTest(Test* test) {
    _tests.push_back(test);
  }


 private:
  TestSuite(DefaultToken) : Test("All Tests"), _default(true) {}
  bool _default;
  std::vector<Test*> _tests;
};
