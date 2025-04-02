//
// Created by PikachuHy on 2023/3/19.
//
#include "pscm/Char.h"
#include "pscm/Port.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <spdlog/fmt/fmt.h>
#include <string>
#include <unicode/uchar.h>
using namespace std::string_literals;

namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Char");
static Char ch_at("@");
static Char ch_star("*");
static Char ch_space(" ");
static Char ch_newline("\n");
static Char ch_add("+");
static Char ch_minus("-");
static Char ch_semicolon(";");
static Char ch_point(".");

Cell Char::from(UChar32 ch) {
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
    return new Char(EOF);
  }
  else if (u_isalnum(ch)) {
    return new Char(ch);
  }
  else {
    return new Char(ch);
  }
}

UString Char::to_string() const {
  switch (ch_) {
  case '\n': {
    return "#\\newline";
  }
  case ' ': {
    return "#\\space";
  }
  default: {
    UString out("#\\");
    out += ch_;
    return out;
  }
  }
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
  port.write_char(ch_);
}

bool Char::is_alphabetic() const {
  return u_isalpha(ch_);
}

bool Char::is_numeric() const {
  return u_isdigit(ch_);
}

bool Char::is_whitespace() const {
  return u_isWhitespace(ch_);
}

bool Char::is_eof() const {
  return ch_ == EOF;
}

Char Char::to_downcase() const {
  return Char(u_tolower(ch_));
}

Char Char::to_upcase() const {
  return Char(u_toupper(ch_));
}

UChar32 Char::to_int() const {
  return int(ch_);
}
} // namespace pscm
