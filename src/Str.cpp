//
// Created by PikachuHy on 2023/3/20.
//

#include "pscm/Str.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
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

bool String::operator<(const String& rhs) const {
  return data_ < rhs.data_;
}

bool String::operator>(const String& rhs) const {
  return data_ > rhs.data_;
}

bool String::operator<=(const String& rhs) const {
  return data_ <= rhs.data_;
}

bool String::operator>=(const String& rhs) const {
  return data_ >= rhs.data_;
}

void String::set(std::size_t idx, char ch) {
  PSCM_ASSERT(idx < data_.size());
  data_[idx] = ch;
}

String String::to_downcase() const {
  std::string str;
  str.resize(data_.size());
  for (size_t i = 0; i < data_.size(); i++) {
    str[i] = std::tolower(data_[i]);
  }
  return String(std::move(str));
}

String String::substring(std::int64_t start, std::int64_t end) const {
  auto s = data_.substr(start, end - start);
  return String(std::move(s));
}

void String::fill(char ch) {
  std::fill(data_.begin(), data_.end(), ch);
}

String operator""_str(const char *data, std::size_t len) {
  return String(std::string(data, len));
}
} // namespace pscm
