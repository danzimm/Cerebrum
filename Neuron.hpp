
#ifndef Neuron_hpp
#define Neuron_hpp

#include <cmath>

template<size_t NInput, size_t NOutput, size_t N, typename Value=double, typename Activator=SigmoidFunction, typename Randomizer=DoubleRandomizer, typename CostFunction=QuadraticCost>
void learn(const std::array<InputNeuron<Value> *, NInput>& inputs,
           const std::array<OutputNeuron<N, Value, Activator, Randomizer> *, NOutput>& outputs,
           size_t ntestcases,
           std::pair<Value *, Value *> *testcases);

struct SigmoidFunction {
  double operator()(double inval) {
    return 1.0 / (1 + std::exp(-inval));
  }
};

struct DoubleRandomizer {
  double operator()() {
    int32_t value = (int32_t)arc4random();
    if (value < 0) {
      return (double)value / (double)INT32_MIN;
    } else {
      return (double)value / (double)INT32_MAX;
    }
  }
};

struct QuadraticCost {
  template<size_t N>
  double operator()(const std::array<double, N>& expected, const std::array<double, N>& values) {
    double result = 0.0;
    for (size_t i = 0; i < N; i++) {
      double tmp = expected[i] - values[i];
      result += tmp * tmp;
    }
    return result / (double)(2 * N);
  }
  
  template<size_t N>
  double prime()(const std::array<double, N>& expected, const std::array<double, N>& values, const std::array<double, N>& primes) {
    double result = 0.0;
    for (size_t i = 0; i < N; i++) {
      double tmp = values[i] - expected[i];
      result += tmp * primes[i];
    }
    return result;
  }
};

private:
  static QuadraticCostPrime<Activator> prime;
};

template<typename Value=double>
struct Neuron {
  Neuron() {}

  virtual Value activate() {
    return _value;
  }

  virtual void decache() {}

protected:
  Value _value;
};

template<typename Value=double>
struct InputNeuron : Neuron<Value> {
  InputNeuron() {}

  void operator()(const Value& inval) {
    _value = inval;
  }

  void operator()(Value&& inval) {
    _value = inval;
  }
};

template<size_t N, typename Value=double, typename Activator=SigmoidFunction, typename Randomizer=DoubleRandomizer>
struct OutputNeuron : Neuron<Value> {
  OutputNeuron() : _bias(DoubleRandomizer()()) {}
  
  virtual Value activate() {
    if (!_cached) {
      Value input = 0;
      for (size_t i = 0; i < _inputCount; i++) {
        auto& weightedNeuron = _inputs[i];
        input += weightedNeuron.first->value() * weightedNeuron.second;
      }
      _value = Activator()(input + _bias);
    }
    return _value;
  }

  virtual void decache() {
    _cached = false;
    _delWeights.fill(0);
    _delBias = 0;
    for (size_t i = 0; i < _inputCount; i++) {
      _inputs[i].first->decache();
    }
  }

  void addInput(Neuron *input) {
    if (_inputCount == N) {
      throw out_of_range("Attempted to add input to output neuron that is full");
    }
    _inputs[_inputCount++] = std::make_tuple(input, DoubleRandomizer()());
  }

  void addToDelWeights(const std::array<Value, N>& delWeights) {
    for (size_t i = 0; i < N; i++) {
      _delWeights[i] += delWeights[i];
    }
  }
  
  void addToDelBias(Value delBias) {
    _delBias += delBias;
  }

protected:
  bool _cached;
  std::array<std::pair<Neuron<Value> *, Value>, N> _inputs;
  Value _bias;
  size_t _inputCount;

  std::array<Value, N> _delWeights;
  Value _delBias;
};

template<size_t NInput, size_t NOutput, size_t N, typename Value=double, typename Activator=SigmoidFunction, typename Randomizer=DoubleRandomizer, typename CostFunction=QuadraticCost>
void learn(const std::array<InputNeuron<Value> *, NInput>& inputs,
           const std::array<OutputNeuron<N, Value, Activator, Randomizer> *, NOutput>& outputs,
           size_t ntestcases,
           std::pair<Value *, Value *> *testcases) {
  for (auto neuron : outputs) {
    neuron->decache();
  }

  for (size_t i = 0; i < ntestcases; i++) {
    std::pair<Value *, Value *>& testcase = testcases[i];
    Value *ins = testcase.first;
    Value *outs = testcase.second;
    // Set up the inputs
    for (size_t j = 0; j < NInput; j++) {
      inputs[i](ins[i]);
    }
    
  }
}
           

#endif // Neuron_hpp

