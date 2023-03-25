//
// Created by PikachuHy on 2023/3/20.
//

#include "pscm/Str.h"
#include <iostream>

namespace pscm {
void String::display() const {
  for (auto ch : data_) {
    std::cout << ch;
  }
}
} // namespace pscm