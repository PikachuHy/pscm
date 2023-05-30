//
// Created by PikachuHy on 2023/2/23.
//

#include "pscm/Exception.h"

#include <iostream>

namespace pscm {
const char *Exception::what() const noexcept {
  print_stack_trace();
  return msg_.c_str();
}

void Exception::print_stack_trace() const {
  std::cout << stack_trace_ << std::endl;
}
} // namespace pscm