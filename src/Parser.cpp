//
// Created by PikachuHy on 2023/2/23.
//

#include "pscm/Parser.h"
#include "pscm/Char.h"
#include "pscm/Exception.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include <cmath>
#include <limits>
#include <string>
using namespace std::string_literals;

namespace pscm {

class NumberParser {
public:
  NumberParser(std::string_view data)
      : data_(data) {
  }

  Number parse() {
    int64_t val{};
    int64_t val2{};
    bool has_point = false;
    while (pos_ < data_.size()) {
      char ch = data_[pos_];
      if (ch == ' ') {
        pos_++;
        continue;
      }
      if (ch == '-') {
        if (negative_) {
          PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
        }
        negative_ = true;
        pos_++;
        continue;
      }
      if (std::isdigit(ch)) {
        if (has_point) {
          val2 = val2 * 10 + (ch - '0');
        }
        else {
          val = val * 10 + (ch - '0');
        }
        pos_++;
        continue;
      }
      if (ch == '.') {
        if (has_point) {
          PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
        }
        has_point = true;
        pos_++;
        continue;
      }
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
    if (has_point) {
      std::string s = std::to_string(val) + "." + std::to_string(val2);
      double v = std::stod(s.data());
      return { v };
    }
    else {
      if (negative_) {
        return { -val };
      }
      return { val };
    }
  }

private:
  std::string_view data_;
  std::size_t pos_{};
  bool has_parsed_ = false;
  bool negative_ = false;
};

Cell Parser::parse() {
  Cell ret{};
  has_parsed_ = false;
  while (!has_parsed_ && pos_ < code_.size()) {
    auto token = next_token();
    switch (token) {
    case Token::LEFT_PARENTHESES: {
      ret = parse_expr();
      has_parsed_ = true;
      break;
    }
    case Token::SYMBOL: {
      ret = Cell(last_symbol_);
      has_parsed_ = true;
      break;
    }
    case Token::NUMBER: {
      ret = Cell(last_num_);
      has_parsed_ = true;
      break;
    }
    case Token::QUOTE: {
      ret = parse();
      ret = cons(quote, cons(ret, nil));
      has_parsed_ = true;
      break;
    }
    case Token::SEMICOLON: {
      skip_line();
      has_parsed_ = false;
      break;
    }
    case Token::QUOTATION: {

      break;
    }
    default: {
      // TODO:
      throw Exception("Unsupported token: ");
    }
    }
  }
  return ret;
}

Cell Parser::parse_expr() {
  Pair *ret = cons(nil, nil);
  auto p = ret;
  while (pos_ < code_.size()) {
    skip_empty();
    auto token = next_token();
    switch (token) {
    case Token::SYMBOL: {
      if (has_dot) {
        p->second = last_symbol_;
        has_dot = false;
      }
      else {
        PSCM_ASSERT(p->second.is_nil());
        auto p2 = cons(Cell(last_symbol_), nil);
        p->second = p2;
        p = p2;
      }
      break;
    }
    case Token::NUMBER: {
      auto p2 = cons(Cell(last_num_), nil);
      p->second = p2;
      p = p2;
      break;
    }
    case Token::SHARP: {
      auto val = parse_literal();
      auto p2 = cons(val, nil);
      p->second = p2;
      p = p2;
      break;
    }
    case Token::QUOTE: {
      auto val = parse();
      auto p2 = cons(cons(quote, cons(val, nil)), nil);
      p->second = p2;
      p = p2;
      break;
    }
    case Token::QUOTATION: {
      auto val = parse_string();
      auto p2 = cons(cons(quote, cons(val, nil)), nil);
      p->second = p2;
      p = p2;
      break;
    }
    case Token::DOT: {
      has_dot = true;
      break;
    }
    case Token::RIGHT_PARENTHESES: {
      return ret->second;
    }
    case Token::LEFT_PARENTHESES: {
      auto expr = parse_expr();
      auto p2 = cons(expr, nil);
      p->second = p2;
      p = p2;
      break;
    }
    default: {
      PSCM_THROW_EXCEPTION("Unsupported token " + std::to_string(int(token)));
    }
    }
  }
  throw Exception("Invalid expr: " + code_);
  return {};
}

Cell Parser::parse_literal() {
  std::string key;
  while (pos_ < code_.size()) {
    if (std::isspace(code_[pos_]) || code_[pos_] == ')') {
      break;
    }
    key += code_[pos_];
    pos_++;
  }
  if (key == "t") {
    return Cell::bool_true();
  }
  if (key == "f") {
    return Cell::bool_false();
  }
  if (key.size() == 2 && key[0] == '\\') {
    return Char::from(key[1]);
  }
  PSCM_THROW_EXCEPTION("Unsupported literal: " + key);
}

Cell Parser::parse_string() {
  auto start = pos_;
  while (pos_ < code_.size() && code_[pos_] != '"') {
    pos_++;
  }
  if (pos_ < code_.size() && code_[pos_] == '"') {
    Cell ret(new String(code_.substr(start, pos_ - start)));
    pos_++;
    return ret;
  }
  PSCM_THROW_EXCEPTION("Invalid string: " + code_.substr(start));
}

void Parser::skip_empty() {
  while (pos_ < code_.size() && std::isspace(code_[pos_])) {
    pos_++;
  }
}

void Parser::skip_line() {
  while (pos_ < code_.size() && code_[pos_] != '\n') {
    pos_++;
  }
  if (pos_ < code_.size() && code_[pos_] == '\n') {
    pos_++;
  }
}

void Parser::eat(char ch) {
  if (code_[pos_] != ch) {
    throw Exception("Invalid code: " + code_.substr(pos_) + "\n" + "Expect: " + ch + "\n" + "Current: " + code_[pos_]);
  }
  pos_++;
}

Parser::Token Parser::next_token() {
  skip_empty();
  if (pos_ >= code_.size()) {
    return Token::NONE;
  }
  char ch = code_[pos_];
  switch (ch) {
  case '(': {
    pos_++;
    return Token::LEFT_PARENTHESES;
  }
  case ')': {
    pos_++;
    return Token::RIGHT_PARENTHESES;
  }
  case '#': {
    pos_++;
    return Token::SHARP;
  }
  case '\'': {
    pos_++;
    return Token::QUOTE;
  }
  case '.': {
    pos_++;
    return Token::DOT;
  }
  case ';': {
    pos_++;
    return Token::SEMICOLON;
  }
  case '"': {
    pos_++;
    return Token::QUOTATION;
  }
  default: {
    if (isdigit(ch) || (ch == '-' && pos_ + 1 < code_.size() && std::isdigit(code_[pos_ + 1]))) {
      int len = 0;
      while (pos_ < code_.size() && !std::isspace(code_[pos_ + len]) && code_[pos_ + len] != ')') {
        len++;
      }
      auto num = NumberParser(std::string_view(code_).substr(pos_, len)).parse();
      last_num_ = new Number(num);
      pos_ += len;
      return Token::NUMBER;
    }
    int len = 0;
    while (pos_ < code_.size() && !std::isspace(code_[pos_ + len]) && code_[pos_ + len] != ')') {
      len++;
    }
    last_symbol_ = new Symbol(std::string_view(code_).substr(pos_, len));
    pos_ += len;
    return Token::SYMBOL;
  }
  }
}

Number operator""_num(const char *data, std::size_t len) {
  NumberParser parser(std::string_view(data, len));
  return parser.parse();
}
} // namespace pscm