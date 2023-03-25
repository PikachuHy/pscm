//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include <exception>
#include <string>

namespace pscm {
class Exception : public std::exception {
public:
  Exception(std::string msg)
      : msg_(std::move(msg)) {
  }

  const char *what() const noexcept override {
    return msg_.c_str();
  }

private:
  std::string msg_;
};

} // namespace pscm