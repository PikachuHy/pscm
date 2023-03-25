//
// Created by PikachuHy on 2023/3/20.
//

#pragma once
#include <string>

namespace pscm {

class String {
public:
  String(std::string data)
      : data_(std::move(data)) {
  }

  void display() const;

private:
  std::string data_;
};

} // namespace pscm
