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

std::ostream& operator<<(std::ostream& os, const String& s) {
  os << '"';
  for (auto ch : s.data_) {
    os << ch;
  }
  os << '"';
  return os;
}

bool String::operator==(const String& rhs) const {
  return data_ == rhs.data_;
}

String operator""_str(const char *data, std::size_t len) {
  return String(std::string(data, len));
}
} // namespace pscm
