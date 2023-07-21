//
// Created by PikachuHy on 2023/2/23.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Exception.h"

#include <iostream>
#endif
namespace pscm {
const char *Exception::what() const noexcept {
  print_stack_trace();
  return msg_.c_str();
}

void Exception::print_stack_trace() const {
#if defined(WASM_PLATFORM) || defined(_MSC_VER) || defined(__ANDROID__)
#else
  std::cout << stack_trace_ << std::endl;
#endif
}
} // namespace pscm
