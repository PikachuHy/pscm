//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include <exception>
#include <string>
#include <ust.hpp>

namespace pscm {
class Exception : public std::exception {
public:
  Exception(std::string msg, ust::StackTrace stack_trace = ust::generate())
      : msg_(std::move(msg))
      , stack_trace_(std::move(stack_trace)) {
  }

  const char *what() const noexcept override;
  void print_stack_trace() const;

private:
  std::string msg_;
  ust::StackTrace stack_trace_;
};

} // namespace pscm