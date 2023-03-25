//
// Created by PikachuHy on 2023/3/19.
//

#pragma once
#include "pscm/Cell.h"
#include <string>

namespace pscm {

class Char {
public:
  Char(std::string ch)
      : ch_(std::move(ch)) {
  }

  static Cell from(char ch);
  friend std::ostream& operator<<(std::ostream& out, const Char& ch);
  void display() const;

private:
  std::string ch_;
};

} // namespace pscm
