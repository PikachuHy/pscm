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

  Char(Char&& ch) {
    ch_ = std::move(ch.ch_);
  }

  static Cell from(char ch);
  bool operator==(const Char& rhs) const;
  bool operator<(const Char& rhs) const;
  bool operator>(const Char& rhs) const;
  bool operator<=(const Char& rhs) const;
  bool operator>=(const Char& rhs) const;
  bool is_alphabetic() const;
  bool is_numeric() const;
  bool is_whitespace() const;
  Char to_downcase() const;
  Char to_upcase() const;
  std::int64_t to_int() const;
  friend std::ostream& operator<<(std::ostream& out, const Char& ch);
  void display() const;

private:
  std::string ch_;
};

} // namespace pscm
