//
// Created by PikachuHy on 2023/3/19.
//

#pragma once
#include "pscm/Cell.h"
#include <string>

namespace pscm {
class Port;

class Char {
public:
  Char(UString ch)
      : ch_(ch.char32At(0)) {
  }

  Char(Char&& ch) {
    ch_ = std::move(ch.ch_);
  }

  Char(UChar32 ch) {
    ch_ = ch;
  }

  static Cell from(UChar32 ch);
  bool operator==(const Char& rhs) const;
  bool operator<(const Char& rhs) const;
  bool operator>(const Char& rhs) const;
  bool operator<=(const Char& rhs) const;
  bool operator>=(const Char& rhs) const;
  bool is_alphabetic() const;
  bool is_numeric() const;
  bool is_whitespace() const;
  bool is_eof() const;
  Char to_downcase() const;
  Char to_upcase() const;
  UChar32 to_int() const;
  UString to_string() const;
  void display(Port& port) const;

private:
  UChar32 ch_;
};

} // namespace pscm
