//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include "pscm/misc/ICUCompat.h"
#include <cstring>
#include <exception>
#include <string>
#if defined(WASM_PLATFORM) || defined(_MSC_VER) || defined(__ANDROID__)
#else
#include <ust/ust.hpp>
#endif
namespace pscm {
class Exception : public std::exception {
public:
#if defined(WASM_PLATFORM) || defined(_MSC_VER) || defined(__ANDROID__)
  Exception(UString msg) {
    msg.toUTF8String(msg_);
  }
#else
  Exception(UString msg, ust::StackTrace stack_trace = ust::generate())
      : stack_trace_(std::move(stack_trace)) {
    msg.toUTF8String(msg_);
  }
#endif
  const char *what() const noexcept override;
  void print_stack_trace() const;

private:
  std::string msg_;
#if defined(WASM_PLATFORM) || defined(_MSC_VER) || defined(__ANDROID__)
#else
  ust::StackTrace stack_trace_;
#endif
};

} // namespace pscm
