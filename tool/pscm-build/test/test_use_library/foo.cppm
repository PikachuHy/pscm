module;
#include <iostream>
export module foo;

export void foo() {
  std::cout << "Hello from foo" << std::endl;
}