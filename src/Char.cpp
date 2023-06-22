//
// Created by PikachuHy on 2023/3/19.
//

#include "pscm/Char.h"
#include "pscm/Port.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <string>
using namespace std::string_literals;

namespace pscm {
static Char ch_at("@");
static Char ch_star("*");
static Char ch_space(" ");
static Char ch_newline("\n");
static Char ch_add("+");
static Char ch_minus("-");
static Char ch_semicolon(";");
static Char ch_point(".");

Cell Char::from(char ch) {
  if (ch == '@') {
    return &ch_at;
  }
  else if (ch == '*') {
    return &ch_star;
  }
  else if (ch == ' ') {
    return &ch_space;
  }
  else if (ch == '\n') {
    return &ch_newline;
  }
  else if (ch == '+') {
    return &ch_add;
  }
  else if (ch == '-') {
    return &ch_minus;
  }
  else if (ch == ';') {
    return &ch_semicolon;
  }
  else if (ch == '.') {
    return &ch_point;
  }
  else if (ch == '?') {
    static Char tmp("?");
    return &tmp;
  }
  else if (ch == '(') {
    static Char tmp("(");
    return &tmp;
  }
  else if (ch == '#') {
    static Char tmp("#");
    return &tmp;
  }
  else if (ch == '~') {
    static Char tmp("~");
    return &tmp;
  }
  else if (ch == '%') {
    static Char tmp("%");
    return &tmp;
  }
  else if (ch == '=') {
    static Char tmp("=");
    return &tmp;
  }
  else if (ch == '/') {
    static Char tmp("/");
    return &tmp;
  }
  else if (ch == ':') {
    static Char tmp(":");
    return &tmp;
  }
  else if (ch == '&') {
    static Char tmp("&");
    return &tmp;
  }
  else if (ch == '<') {
    static Char tmp("<");
    return &tmp;
  }
  else if (ch == '>') {
    static Char tmp(">");
    return &tmp;
  }
  else if (ch == EOF) {
    std::string s;
    s.resize(1);
    s[0] = ch;
    return new Char(std::move(s));
  }
  else if (std::isalnum(ch)) {
    std::string s;
    s.resize(1);
    s[0] = ch;
    return new Char(std::move(s));
  }
  else if (int(ch) <= 32) {
    std::string s;
    s.resize(1);
    s[0] = ch;
    return new Char(std::move(s));
  }
  else {
    PSCM_THROW_EXCEPTION("unsupported char: "s + ch);
  }
}

std::ostream& operator<<(std::ostream& out, const Char& ch) {
  PSCM_ASSERT(!ch.ch_.empty());
  if (ch.ch_.size() == 1) {
    switch (ch.ch_.at(0)) {
    case '\n': {
      out << "#\\newline";
      return out;
    }
    case ' ': {
      out << "#\\space";
      return out;
    }
    }
  }
  return out << "#\\" << ch.ch_;
}

bool Char::operator==(const Char& rhs) const {
  return ch_ == rhs.ch_;
}

bool Char::operator<(const Char& rhs) const {
  return ch_ < rhs.ch_;
}

bool Char::operator>(const Char& rhs) const {
  return ch_ > rhs.ch_;
}

bool Char::operator<=(const Char& rhs) const {
  return ch_ <= rhs.ch_;
}

bool Char::operator>=(const Char& rhs) const {
  return ch_ >= rhs.ch_;
}

void Char::display(Port& port) const {
  PSCM_ASSERT(!ch_.empty());
  for (auto ch : ch_) {
    port.write_char(ch);
  }
}

bool Char::is_alphabetic() const {
  return ch_.size() == 1 && std::isalpha(ch_[0]);
}

bool Char::is_numeric() const {
  return ch_.size() == 1 && std::isalnum(ch_[0]) && !std::isalpha(ch_[0]);
}

bool Char::is_whitespace() const {
  return ch_.size() == 1 && std::isspace(ch_[0]);
}

bool Char::is_eof() const {
  return ch_.size() == 1 && ch_[0] == EOF;
}

Char Char::to_downcase() const {
  std::string str;
  str.resize(ch_.size());
  for (size_t i = 0; i < ch_.size(); i++) {
    str[i] = std::tolower(ch_[i]);
  }
  return Char(std::move(str));
}

Char Char::to_upcase() const {
  std::string str;
  str.resize(ch_.size());
  for (size_t i = 0; i < ch_.size(); i++) {
    str[i] = std::toupper(ch_[i]);
  }
  return Char(std::move(str));
}

std::int64_t Char::to_int() const {
  PSCM_ASSERT(ch_.size() == 1);
  return int(ch_[0]);
}
} // namespace pscm
