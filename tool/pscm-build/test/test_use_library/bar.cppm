module;
#include <iostream>
export module bar;

export void bar() {
  std::cout << "Hello from foo" << std::endl;
}