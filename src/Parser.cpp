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
#include "pscm/scm_utils.h"
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
    auto start = pos_;
    auto token = next_token();
    ret = parse_token(token, start);
  }
  return ret;
}

Cell Parser::parse_token(pscm::Parser::Token token, std::size_t start) {
  Cell ret;
  has_parsed_ = false;
  while (!has_parsed_) {
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
      static auto quote = "quote"_sym;
      ret = parse();
      ret = list(&quote, ret);
      has_parsed_ = true;
      break;
    }
    case Token::UNQUOTE: {
      static auto unquote = "unquote"_sym;
      ret = parse();
      ret = list(&unquote, ret);
      has_parsed_ = true;
      break;
    }
    case Token::UNQUOTE_SPLICING: {
      static auto unquote_splicing = "unquote-splicing"_sym;
      ret = parse();
      ret = list(&unquote_splicing, ret);
      has_parsed_ = true;
      break;
    }
    case Token::SEMICOLON: {
      skip_line();
      has_parsed_ = false;
      start = pos_;
      token = next_token();
      break;
    }
    case Token::QUOTATION: {
      ret = parse_string();
      has_parsed_ = true;
      break;
    }
    case Token::QUASIQUOTE: {
      static auto quasiquote = "quasiquote"_sym;
      ret = parse();
      ret = list(&quasiquote, ret);
      has_parsed_ = true;
      break;
    }
    case Token::SHARP: {
      ret = parse_literal();
      has_parsed_ = true;
      break;
    }
    default: {
      // TODO:
      PSCM_THROW_EXCEPTION("Unsupported token: " + code_.substr(start, pos_ - start));
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
    auto start = pos_;
    auto token = next_token();
    switch (token) {
    case Token::RIGHT_PARENTHESES: {
      return ret->second;
    }
    case Token::DOT: {
      auto expr = parse();
      token = next_token();
      if (token == Token::RIGHT_PARENTHESES) {
        p->second = expr;
        return ret->second;
      }
      PSCM_THROW_EXCEPTION("Invalid expr with . : " + code_.substr(start, pos_ - start));
    }
    default: {
      auto expr = parse_token(token, start);
      auto new_pair = cons(expr, nil);
      p->second = new_pair;
      p = new_pair;
    }
    }
  }
  PSCM_THROW_EXCEPTION("Invalid expr: " + code_);
}

Cell Parser::parse_literal() {
  std::string key;
  auto token = next_token();
  switch (token) {
  case Token::SYMBOL: {
    if (last_symbol_->name() == "t") {
      return Cell::bool_true();
    }
    else if (last_symbol_->name() == "f") {
      return Cell::bool_false();
    }
    else {
      PSCM_THROW_EXCEPTION("Unsupported literal: " + std::string(last_symbol_->name()));
    }
    break;
  }
  case Token::BACK_SLASH: {
    // read char
    auto start = pos_;
    auto tok = next_token();
    if (tok != Token::SYMBOL) {
      PSCM_THROW_EXCEPTION("Invalid char: " + code_.substr(start, pos_ - start));
    }
    auto key = last_symbol_->name();
    if (key.size() == 1) {
      return Char::from(key[0]);
    }
    else {
      PSCM_THROW_EXCEPTION("Unsupported literal: " + std::string(key));
    }
  }
  case Token::LEFT_PARENTHESES: {
    // read vector constant
    auto expr = parse_expr();
    Cell::Vec vec;
    while (!expr.is_nil()) {
      auto e = car(expr);
      expr = cdr(expr);
      vec.push_back(e);
    }
    return { new Cell::Vec(std::move(vec)) };
  }
  default: {
  }
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
  case ',': {
    if (pos_ + 1 < code_.size() && code_[pos_ + 1] == '@') {
      pos_++;
      pos_++;
      return Token::UNQUOTE_SPLICING;
    }
    else {
      pos_++;
      return Token::UNQUOTE;
    }
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
  case '`': {
    pos_++;
    return Token::QUASIQUOTE;
  }
  case '\\': {
    pos_++;
    return Token::BACK_SLASH;
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