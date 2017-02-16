
#ifndef List_hpp
#define List_hpp

#include <functional>
#include <iostream>

template<typename T>
struct List {
  T value;
  List *next;

  List(const T& val) : value(val), next(NULL) {}
  List(T&& val) : value(val), next(NULL) {}

  static void insert(List *&head, List *node) {
    head = head->insert(node);
  }

  static void insert(List *&head, const T& value) {
    head = head->insert(value);
  }

  static void insert(List *&head, T&& value) {
    head = head->insert(value);
  }

  static void reverse(List *&head) {
    head = head->reverse();
  }

  List *insert(List *node) {
    if (node->next) {
      std::cerr << "Inserting a node that already contains a next. Overwriting." << std::endl;
    }
    node->next = this;
    return node;
  }

  List *insert(const T& val) {
    return insert(new List(val));
  }

  List *insert(T&& val) {
    return insert(new List(val));
  }

  List *reverse() {
    List *previous = NULL;
    List *current = this;
    while (current) {
      List *tmp = current->next;
      current->next = previous;
      previous = current;
      current = tmp;
    }
    return previous;
  }

  void iterate(std::function<void(T&)> callback) {
    List *node = this;
    while (node) {
      callback(node->value);
      node = node->next;
    }
  }

  void iterate(std::function<void(T&, bool&)> callback) {
    List *node = this;
    bool stop = false;
    while (node && !stop) {
      callback(node->value, stop);
      node = node->next;
    }
  }

  uintptr_t length() {
    uintptr_t result = 0;
    List *node = this;
    while (node) {
      result += 1;
      node = node->next;
    }
    return result;
  }
};

#endif

