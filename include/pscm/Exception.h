//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include <cstring>
#include <exception>
#include <string>
#if defined(WASM_PLATFORM) || defined(_MSC_VER)
#else
#include <ust.hpp>
#endif
namespace pscm {
class Exception : public std::exception {
public:
#if defined(WASM_PLATFORM) || defined(_MSC_VER)
  Exception(std::string msg, ust::StackTrace stack_trace = ust::generate())
      : msg_(std::move(msg))
      , stack_trace_(std::move(stack_trace)) {
  }
#else
  Exception(std::string msg)
      : msg_(std::move(msg)) {
  }
#endif
  const char *what() const noexcept override;
  void print_stack_trace() const;

private:
  std::string msg_;
#if defined(WASM_PLATFORM) || defined(_MSC_VER)
#else
  ust::StackTrace stack_trace_;
#endif
};

} // namespace pscm