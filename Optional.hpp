
#ifndef Optional_hpp
#define Optional_hpp

#include <functional>

struct OptionalEmpty {};
static OptionalEmpty OptionalEmptyValue;

template<typename T>
struct Optional {
  T *value;

  Optional(const Optional& opt) : value(opt.value), storage(opt.value ? Storage(opt.storage.value) : Storage(OptionalEmptyValue)) {
    if (value) {
      value = &opt.storage.value;
    }
  }
  Optional(Optional&& opt) : value(opt.value), storage(opt.value ? Storage(opt.storage.value) : Storage(OptionalEmptyValue)) {
    if (value) {
      value = &opt.storage.value;
    }
  }
  Optional(const T& val) : storage(val), value(&storage.value) {}
  Optional(T&& val) : value(NULL), storage(std::move(val)) {
    value = &storage.value;
  }
  Optional() : value(NULL), storage(OptionalEmptyValue) {}

  T getOrElse(std::function<T(void)> callback) {
    if (value) {
      return *value;
    }
    return callback();
  }

  template<typename U>
  Optional<U> map(std::function<U(T&)> func) {
    if (!value) {
      return Optional<U>();
    }
    return Optional<U>(func(*value));
  }

  ~Optional() {
    if (value) {
      storage.value.T::~T();
    }
  }

private:
  union Storage {
    T value;
    char empty;
    
    Storage(OptionalEmpty) : empty(0x13) {}
    
    template<typename ...Args>
    Storage(const Args&... args) : value(args...) {}
    template<typename ...Args>
    Storage(Args&&... args) : value(std::forward<Args>(args)...) {}

    Storage(const Storage& other) : value(other.value) {}
    Storage(Storage&& other) : value(std::forward<T>(other.value)) {}
    
    ~Storage() {}
  } storage;
};

#endif // Optional_hpp
