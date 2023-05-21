//
// Created by PikachuHy on 2023/2/23.
//

#include "pscm/Parser.h"
#include "pscm/Char.h"
#include "pscm/Exception.h"
#include "pscm/Keyword.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
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
      if (data_[pos_] == ' ') {
        pos_++;
        continue;
      }
      break;
    }
    if (pos_ < data_.size() && data_[pos_] == 'i') {
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
    if (pos_ < data_.size() && data_[data_.size() - 1] == 'i') {
      return parse_complex();
    }
    if (data_.find('/') != std::string_view::npos) {
      auto num1_opt = parse_digit();
      if (data_[pos_] != '/') {
        PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
      }
      pos_++;
      auto num2_opt = parse_digit();
      return Rational(num1_opt.value(), num2_opt.value());
    }
    auto sign = parse_sign(true).value_or(false);
    auto num_opt = parse_num();
    if (!num_opt.has_value()) {
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }

    auto num = num_opt.value();
    if (sign) {
      if (num.is_int()) {
        return -num.to_int();
      }
      if (num.is_float()) {
        return -num.to_float();
      }
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
    if (pos_ != data_.size()) {
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
    return num;
  }

  std::optional<std::int64_t> parse_digit(bool optional = false) {
    int count = 0;
    std::int64_t ret = 0;
    while (pos_ < data_.size() && is_digit()) {
      ret = ret * 10 + (data_[pos_] - '0');
      pos_++;
      count++;
    }
    if (count == 0) {
      if (optional) {
        return std::nullopt;
      }
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
    return ret;
  }

  std::optional<Number> parse_num(bool optional = false) {
    auto pos = pos_;
    auto num = parse_digit(optional);
    if (!num.has_value()) {
      if (optional) {
        return std::nullopt;
      }
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
    bool has_point = false;
    std::int64_t point_num = 0;
    if (pos_ < data_.size() && data_[pos_] == '.') {
      has_point = true;
      has_point = true;
      pos_++;
      point_num = parse_digit().value();
    }
    double val = num.value() * 1.0;
    bool has_e = false;
    std::int64_t e_num = 0;
    if (data_[pos_] == 'e' || data_[pos_] == 'E') {
      has_e = true;
      pos_++;
      auto sign = parse_sign(true);
      e_num = parse_digit().value();
      if (sign.value_or(false)) {
        e_num = -e_num;
      }
    }
    auto new_pos = pos_;
    if (has_point || has_e) {
      return convert_str_to_float(std::string(data_.substr(pos, new_pos - pos)));
    }
    else {
      return num;
    }
  }

  Number parse_complex() {
    Number ret;
    auto sign1_opt = parse_sign(true);
    if (has_sign_after(pos_)) {
      auto num1_opt = parse_num();
      auto sign2_opt = parse_sign(false);
      double num1 = num1_opt.value().is_int() ? num1_opt.value().to_int() : num1_opt.value().to_float();
      auto sign1 = sign1_opt.value_or(false);
      if (sign1) {
        num1 = -num1;
      }
      auto num2_tmp = parse_num(true).value_or(1.0);
      double num2 = num2_tmp.is_int() ? num2_tmp.to_int() : num2_tmp.to_float();
      auto sign2 = sign2_opt.value();
      if (sign2) {
        num2 = -num2;
      }
      ret = Complex(num1, num2);
    }
    else {
      if (!sign1_opt.has_value()) {
        PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
      }
      auto num_opt = parse_num(true);
      auto num_tmp = num_opt.value_or(1.0);
      double num = num_tmp.is_int() ? num_tmp.to_int() : num_tmp.to_float();
      auto sign1 = sign1_opt.value();
      if (sign1) {
        num = -num;
      }
      ret = Complex(0, num);
    }
    if (data_[pos_] != 'i') {
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
    return ret;
  }

  std::optional<bool> parse_sign(bool optional) {
    if (data_[pos_] == '-') {
      pos_++;
      return true;
    }
    else if (data_[pos_] == '+') {
      pos_++;
      return false;
    }
    else if (optional) {
      return std::nullopt;
    }
    else {
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
  }

  bool has_sign_after(std::size_t pos) {
    while (pos < data_.size()) {
      if (is_sign(pos)) {
        return true;
      }
      pos++;
    }
    return false;
  }

  bool is_sign(std::size_t pos) {
    return data_[pos] == '+' || data_[pos] == '-';
  }

  bool is_digit() {
    PSCM_ASSERT(pos_ < data_.size());
    return std::isdigit(data_[pos_]);
  }

  double convert_str_to_float(std::string str) {
    SPDLOG_INFO("str: {}", str);
    errno = 0;
    char *end;
    double x = std::stod(str);
    if (errno == ERANGE) {
      if (!(x != 0 && x > -HUGE_VAL && x < HUGE_VAL)) {
        PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
      }
      else {
        return x;
      }
    }
    else if (errno) {
      PSCM_THROW_EXCEPTION("Invalid Number: "s + std::string(data_));
    }
    else {
      return x;
    }
  }

private:
  std::string_view data_;
  std::size_t pos_{};
  bool has_parsed_ = false;
  bool negative_ = false;
  bool is_float = false;
  bool is_complex = false;
};

Parser::Parser(std::string code)
    : code_(std::move(code)) {
}

Parser::Parser(std::string code, std::string_view filename)
    : code_(std::move(code))
    , filename_(filename) {
  int start = 0;
  int offset = 0;
  is_file_ = true;
  while (start + offset < code_.size()) {
    if (code_[start + offset] == '\n') {
      lines_.emplace_back(code_.data() + start, offset);
      start += offset + 1;
      offset = 0;
    }
    else {
      offset++;
    }
  }
}

Parser::Parser(std::istream *in)
    : in_(in)
    , use_stream_(true) {
}

Cell Parser::parse() {
  Cell ret{};
  has_parsed_ = false;
  while (!has_parsed_ && !is_eof()) {
    auto start = pos_;
    auto token = next_token();
    ret = parse_token(token, start);
  }
  return ret;
}

Cell Parser::next() {
  return parse();
}

Cell Parser::parse_token(pscm::Parser::Token token, std::size_t start) {
  Cell ret;
  has_parsed_ = false;
  while (!has_parsed_) {
    switch (token) {
    case Token::END_OF_FILE: {
      has_parsed_ = true;

      break;
    }
    case Token::LEFT_PARENTHESES: {
      ret = parse_expr();
      has_parsed_ = true;
      break;
    }
    case Token::SYMBOL: {
      PSCM_ASSERT(last_symbol_);
      PSCM_ASSERT(!last_symbol_->name().empty());
      if (last_symbol_->name()[0] == ':') {
        ret = Cell(new Keyword(last_symbol_));
      }
      else {
        ret = Cell(last_symbol_);
      }
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
      if (is_file_) {
        PSCM_ASSERT(row_ < lines_.size());
        std::cout << "Parser error occured in " << filename_ << ":" << row_ << std::endl;
        std::cout << lines_[row_] << std::endl;
        for (size_t i = 0; i < col_; i++) {
          std::cout << " ";
        }
        std::cout << "^" << std::endl;
      }
      PSCM_THROW_EXCEPTION("Unsupported token: " + code_.substr(start, pos_ - start));
    }
    }
  }
  return ret;
}

Cell Parser::parse_expr() {
  Pair *ret = cons(nil, nil);
  auto p = ret;
  while (!is_eof()) {
    skip_empty();
    auto start = pos_;
    auto token = next_token();
    switch (token) {
    case Token::SEMICOLON: {
      skip_line();
      break;
    }
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
      auto name = last_symbol_->name();
      if (name.empty()) {
        PSCM_THROW_EXCEPTION("Unsupported literal: " + std::string(last_symbol_->name()));
      }
      if (name[0] == 'e') {
        // parse number literal
        auto val = std::stod(std::string(name.substr(1)));
        return new Number(val);
      }
      if (name[0] == 'i') {
        NumberParser parser(name.substr(1));
        auto val = parser.parse();
        return new Number(val);
      }
      PSCM_THROW_EXCEPTION("Unsupported literal: " + std::string(last_symbol_->name()));
    }
    break;
  }
  case Token::BACK_SLASH: {
    // hack: #\ = #\Space
    if (peek_char() == ' ') {
      next_char();
      return Char::from(' ');
    }
    // read char
    auto start = pos_;
    auto tok = next_token();
    if (tok == Token::DOT) {
      return Char::from('.');
    }
    else if (tok == Token::SYMBOL) {
      auto key = last_symbol_->name();
      if (key.size() == 1) {
        return Char::from(key[0]);
      }
      else if (key == "space" || key == "Space") {
        return Char::from(' ');
      }
      else if (key == "newline") {
        return Char::from('\n');
      }
      else if (key == "return") {
        return Char::from(13);
      }
      else if (key == "ht") {
        return Char::from(9);
      }
      else if (key == "lf") {
        return Char::from(10);
      }
      else if (key == "vt") {
        return Char::from(11);
      }
      else if (key == "ff") {
        return Char::from(12);
      }
      else {
        PSCM_THROW_EXCEPTION("Unsupported literal: " + std::string(key));
      }
    }
    else if (tok == Token::SEMICOLON) {
      return Char::from(';');
    }
    else if (tok == Token::NUMBER) {
      PSCM_ASSERT(last_num_);
      PSCM_ASSERT(last_num_->is_int());
      auto n = last_num_->to_int();
      if (n >= 0 && n < 10) {
        return Char::from('0' + n);
      }
      PSCM_THROW_EXCEPTION("Unsupported literal: " + std::to_string(n));
    }
    else {
      if (key.empty()) {
        return Char::from(' ');
      }
      PSCM_THROW_EXCEPTION("Invalid char: " + code_.substr(start, pos_ - start));
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
  std::string s;
  while (!is_eof()) {
    if (peek_char() == '"') {
      next_char();
      break;
    }
    if (peek_char() == '\\') {
      next_char();
      if (is_eof()) {
        PSCM_THROW_EXCEPTION("Invalid String: ");
      }
      auto ch = peek_char();
      switch (ch) {
      case 't': {
        s.push_back('\t');
        break;
      }
      case 'n': {
        s.push_back('\n');
        break;
      }
      case '"': {
        s.push_back('"');
        break;
      }
      case '\\': {
        s.push_back('\\');
        break;
      }
      default: {
        PSCM_THROW_EXCEPTION("Unsupported char \\"s + ch + ", current string: " + s);
      }
      }
      next_char();
    }
    else {
      s.push_back(next_char());
    }
  }
  Cell ret(new String(s));
  return ret;
  // while (pos_ < code_.size()) {
  //   if (code_[pos_] == '"') {
  //     break;
  //   }
  //   if (code_[pos_] == '\\') {
  //     if (pos_ + 1 < code_.size()) {
  //       advance();
  //     }
  //     else {
  //       PSCM_THROW_EXCEPTION("Invalid String: " + code_.substr(start));
  //     }
  //   }
  //   s.push_back(code_[pos_]);
  //   advance();
  // }
  // if (pos_ < code_.size() && code_[pos_] == '"') {
  //   Cell ret(new String(s));
  //   advance();
  //   return ret;
  // }
  // PSCM_THROW_EXCEPTION("Invalid string: " + code_.substr(start));
}

void Parser::skip_empty() {
  while (!is_eof()) {
    char ch = peek_char();
    if (!std::isspace(ch)) {
      break;
    }
    ch = next_char();
  }
}

void Parser::skip_line() {
  while (!is_eof()) {
    char ch = peek_char();
    next_char();
    if (ch == '\n') {
      break;
    }
  }
}

void Parser::eat(char ch) {
  if (code_[pos_] != ch) {
    throw Exception("Invalid code: " + code_.substr(pos_) + "\n" + "Expect: " + ch + "\n" + "Current: " + code_[pos_]);
  }
  advance();
}

Parser::Token Parser::next_token() {
  skip_empty();
  if (is_eof()) {
    if (is_file_ || use_stream_) {
      return Token::END_OF_FILE;
    }
    return Token::NONE;
  }
  char ch = next_char();
  switch (ch) {
  case '(': {
    return Token::LEFT_PARENTHESES;
  }
  case ')': {
    return Token::RIGHT_PARENTHESES;
  }
  case '#': {
    return Token::SHARP;
  }
  case '\'': {
    return Token::QUOTE;
  }
  case ',': {
    if (peek_char() == '@') {
      next_char();
      return Token::UNQUOTE_SPLICING;
    }
    else {
      return Token::UNQUOTE;
    }
  }
  case ';': {
    return Token::SEMICOLON;
  }
  case '"': {
    return Token::QUOTATION;
  }
  case '`': {
    return Token::QUASIQUOTE;
  }
  case '\\': {
    return Token::BACK_SLASH;
  }
  case '.': {
    if (peek_char() == ' ') {
      return Token::DOT;
    }
  }
  default: {
    auto row = row_ + 1;
    auto col = col_;
    std::string s;
    s.push_back(ch);
    read_until(s, "()\"'`,;");
    if (std::isdigit(ch) || (s.size() > 1 && (ch == '-' || ch == '+'))) {
      try {
        auto num = NumberParser(s).parse();
        last_num_ = new Number(num);
        return Token::NUMBER;
      }
      catch (...) {
      }
    }
    last_symbol_ = new Symbol(s, filename_, row, col);
    return Token::SYMBOL;
  }
  }
}

void Parser::advance() {
  PSCM_ASSERT(pos_ <= code_.size());
  if (code_[pos_] == '\n') {
    row_++;
    col_ = 0;
  }
  pos_++;
  col_++;
}

bool Parser::is_eof() const {
  if (use_stream_) {
    PSCM_ASSERT(in_);
    return in_->eof();
  }
  else {
    return pos_ >= code_.size();
  }
}

char Parser::next_char() {
  if (use_stream_) {
    char ch;
    in_->get(ch);
    return ch;
  }
  else {
    char ch = code_[pos_];
    advance();
    return ch;
  }
}

char Parser::peek_char() {
  if (use_stream_) {
    char ch;
    ch = in_->peek();
    return ch;
  }
  else {
    if (is_eof()) {
      return EOF;
    }
    else {
      char ch = code_[pos_];
      return ch;
    }
  }
}

void Parser::read_until(std::string& s, std::string_view end) {
  char ch;
  while (!is_eof()) {
    ch = peek_char();
    if (std::isspace(ch)) {
      break;
    }
    if (end.find(ch) != std::string_view::npos) {
      break;
    }
    s.push_back(ch);
    ch = next_char();
  }
}

Number operator""_num(const char *data, std::size_t len) {
  NumberParser parser(std::string_view(data, len));
  return parser.parse();
}
} // namespace pscm
