//
// Created by PikachuHy on 2023/2/23.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Parser.h"
#include "pscm/Char.h"
#include "pscm/Exception.h"
#include "pscm/Keyword.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Port.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include "unicode/uchar.h"
#include "unicode/ustream.h"
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#endif
using namespace std::string_literals;

namespace pscm {

PSCM_INLINE_LOG_DECLARE("pscm.core.Parser");

class NumberParser {
public:
  NumberParser(const UString& data, SourceLocation loc = {})
      : data_(data)
      , loc_(loc)
      , iter_(data) {
  }

  Number parse() {
    while (iter_.hasNext() && (iter_.current32() == ' ')) {
      iter_.next32PostInc();
    }
    if (iter_.hasNext() && iter_.current32() == 'i') {
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
    }
    if (iter_.hasNext() && data_.char32At(data_.length() - 1) == 'i') {
      return parse_complex();
    }
    if (data_.indexOf('/') != -1) {
      auto num1_opt = parse_digit();
      if (iter_.next32PostInc() != '/') {
        PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
      }
      auto num2_opt = parse_digit();
      return Rational(num1_opt.value(), num2_opt.value());
    }
    auto sign = parse_sign(true).value_or(false);
    auto num_opt = parse_num();
    if (!num_opt.has_value()) {
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
    }

    auto num = num_opt.value();
    if (sign) {
      if (num.is_int()) {
        return -num.to_int();
      }
      if (num.is_float()) {
        return -num.to_float();
      }
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
    }
    if (iter_.getIndex() != iter_.endIndex()) {
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
    }
    return num;
  }

  std::optional<std::int64_t> parse_digit(bool optional = false) {
    int count = 0;
    std::int64_t ret = 0;
    while (iter_.hasNext() && is_digit()) {
      
      ret = ret * 10 + u_digit(iter_.next32PostInc(), 10);
      count++;
    }
    if (count == 0) {
      if (optional) {
        return std::nullopt;
      }
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
    }
    return ret;
  }

  std::optional<Number> parse_num(bool optional = false) {
    auto pos = iter_.getIndex();
    auto num = parse_digit(optional);
    if (!num.has_value()) {
      if (optional) {
        return std::nullopt;
      }
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
    }
    bool has_point = false;
    if (iter_.hasNext() && iter_.current32() == '.') {
      has_point = true;
      has_point = true;
      iter_.next32PostInc();
      parse_digit().value();
    }
    bool has_e = false;
    std::int64_t e_num = 0;
    if (iter_.hasNext() && (iter_.current32() == 'e' || iter_.current32() == 'E')) {
      has_e = true;
      iter_.next32PostInc();
      auto sign = parse_sign(true);
      e_num = parse_digit().value();
      if (sign.value_or(false)) {
        e_num = -e_num;
      }
    }
    auto new_pos = iter_.getIndex();
    if (has_point || has_e) {
      UString numstr;
      data_.extractBetween(pos, new_pos, numstr);
      return convert_str_to_float(numstr);
    }
    else {
      return num;
    }
  }

  Number parse_complex() {
    Number ret;
    auto sign1_opt = parse_sign(true);
    if (has_sign_after(iter_)) {
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
        PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
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
    if (iter_.current32() != 'i') {
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
    }
    return ret;
  }

  std::optional<bool> parse_sign(bool optional) {
    if (!iter_.hasNext()) {
      return std::nullopt;
    }
    auto ch = iter_.current32();
    if (ch == '-') {
      iter_.next32PostInc();
      return true;
    }
    else if (ch == '+') {
      iter_.next32PostInc();
      return false;
    }
    else if (optional) {
      return std::nullopt;
    }
    else {
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
    }
  }

  bool has_sign_after(UIterator& iter) {
    while (iter.hasNext()) {
      if (is_sign(iter)) {
        return true;
      }
      iter.next32PostInc();
    }
    return false;
  }

  bool is_sign(const UIterator& iter) {
    return iter.current32() == '+' || iter.current32() == '-';
  }

  bool is_digit() {
    PSCM_ASSERT(iter_.hasNext());
    return u_isdigit(iter_.current32());
  }

  double convert_str_to_float(const UString& str) {
    PSCM_INFO("str: {0}", str);
    errno = 0;
    double x;
    auto res = double_from_string(str);
    if (std::holds_alternative<double>(res)) {
      return std::get<double>(res);
    }
    else{
      PSCM_THROW_EXCEPTION(loc_.to_string() + ", Invalid Number: " + data_);
      return 0.0;
    }
  }

private:
  UString data_;
  UIterator iter_;
  SourceLocation loc_;
};

Parser::Parser(const UString & code)
    : code_(code) {
}

Parser::Parser(UIteratorP in)
    : code_(in) {
}

Parser::Parser(const UString & code, const UString& filename)
    : code_(code)
    , filename_(filename) {
  int start = 0;
  is_file_ = true;
  auto iter_ = std::get<UIterator>(code_);
  UChar32 ch;
  while (iter_.hasNext()) {
    ch = iter_.next32();
    if (ch == '\n') { // TODO: Line break Category
      auto end = iter_.getIndex();
      lines_.emplace_back(code, start, end - start);
      start = end;
    }
  }
  iter_.setToStart();
}

Parser::Parser(Port *in)
    : code_(in) {
}

Cell Parser::parse() {
  Cell ret{};
  has_parsed_ = false;
  while (!has_parsed_) {
    last_token_.remove();
    auto token = next_token();
    if (token == Token::NONE) {
      return Cell::none();
    }
    
    ret = parse_token(token);
  }
  return ret;
}

Cell Parser::next() {
  return parse();
}

Cell Parser::parse_token(pscm::Parser::Token token) {
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
      PSCM_ASSERT(!last_symbol_->name().isEmpty());
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
      last_token_.remove();
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
      PSCM_ERROR("Unsupported token: {0}", int(token));
      PSCM_THROW_EXCEPTION("Unsupported token: " + last_token_);
    }
    }
  }
  return ret;
}

Cell Parser::parse_expr() {
  Pair *ret = cons(Cell::nil(), Cell::nil());
  auto p = ret;
  while (true) {
    skip_empty();
    last_token_.remove();
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
      PSCM_THROW_EXCEPTION("Invalid expr with . : " + last_token_);
    }
    case Token::END_OF_FILE: {
      PSCM_THROW_EXCEPTION("Invalid expr: " + last_token_);
    }
    default: {
      auto expr = parse_token(token);
      auto new_pair = cons(expr, nil);
      p->second = new_pair;
      p = new_pair;
    }
    }
  }
}

Cell Parser::parse_literal() {
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
      if (name.isEmpty()) {
        PSCM_THROW_EXCEPTION("Unsupported literal: " + last_symbol_->name());
      }
      if (name[0] == 'e') {
        // parse number literal
        std::string utf8;
        utf8.reserve(name.length());
        name.extract(1, name.length() - 1, utf8.data(), name.length());
        auto val = std::stod(utf8);
        return new Number(val);
      }
      if (name[0] == 'i') {
        UString numstr;
        name.extractBetween(1, name.length(), numstr);
        NumberParser parser(numstr);
        auto val = parser.parse();
        return new Number(val);
      }
      PSCM_THROW_EXCEPTION("Unsupported literal: " + last_symbol_->name());
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
    last_token_.remove();
    auto tok = next_token();
    if (tok == Token::DOT) {
      return Char::from('.');
    }
    else if (tok == Token::SYMBOL) {
      auto key = last_symbol_->name();
      static std::unordered_map<UString, int> literal_map{
        {"nul",  0},
        {"soh",  1},
        {"stx",  2},
        {"etx",  3},
        {"eot",  4},
        {"enq",  5},
        {"ack",  6},
        {"bel",  7},
        { "bs",  8},
        { "ht",  9},
        { "lf", 10},
        { "vt", 11},
        { "ff", 12},
        { "cr", 13},
        { "so", 14},
        { "si", 15},
        {"dle", 16},
        {"dl1", 17},
        {"dc2", 18},
        {"dc3", 19},
        {"dc4", 20},
        {"nak", 21},
        {"syn", 22},
        {"etb", 23},
        {"can", 24},
        { "em", 25},
        {"sub", 26},
        {"esc", 27},
        { "fs", 28},
        { "gs", 29},
        { "rs", 30},
        { "us", 31},
        { "sp", 32},
      };
      auto it = literal_map.find(key);
      if (it != literal_map.end()) {
        return Char::from(it->second);
      }
      if (key.countChar32() == 1) {
        return Char::from(key.char32At(0));
      }
      else if (key == "space" || key == "Space") {
        return Char::from(' ');
      }
      else if (key == "newline") {
        return Char::from('\n');
      }
      else if (key == "return" || key == "cr") {
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
        PSCM_THROW_EXCEPTION("Unsupported literal: " + key);
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
      PSCM_THROW_EXCEPTION("Unsupported literal: " + pscm::to_string(n));
    }
    else {
      if (last_token_.isEmpty()) {
        return Char::from(' ');
      }
      PSCM_THROW_EXCEPTION("Invalid char: " + last_token_);
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
    return { new Cell::Vec(move(vec)) };
  }
  default: {
  }
  }
  PSCM_THROW_EXCEPTION("Unsupported literal: " + last_token_);
}

Cell Parser::parse_string() {
  auto row = row_;
  // auto col = col_;
  // auto start = pos_;
  UString s;
  while (true) {
    auto ch = peek_char();
    if (ch == '"') {
      next_char();
      break;
    } else if (ch == EOF)
    {
      break;
    }
    
    if (ch == '\\') {
      next_char();
      ch = next_char();
      if (ch == EOF) {
        PSCM_THROW_EXCEPTION("Invalid String: ");
      }
      switch (ch) {
      case 't': {
        s += '\t';
        break;
      }
      case 'n': {
        s += '\n';
        break;
      }
      case '"': {
        s += '"';
        break;
      }
      case '\\': {
        s += '\\';
        break;
      }
      default: {
        PSCM_THROW_EXCEPTION("Unsupported char \\"_u + ch + ", current string: " + s);
      }
      }
    }
    else {
      s += next_char();
    }
  }
  if (is_file_) {
    std::string utf8;
    filename_.toUTF8String(utf8);
    Cell ret(new String(s), SourceLocation(utf8.c_str(), row));
    return ret;
  }
  else {

    Cell ret(new String(s));
    return ret;
  }
}

void Parser::skip_empty() {
  auto ch = peek_char();
  while (ch != EOF) {
    if (!u_isspace(ch)) {
      break;
    }
    next_char();
    ch = peek_char();
  }
}

void Parser::skip_line() {
  auto ch = peek_char();
  while (ch != EOF) {
    if (ch == '\n') {
      break;
    }
    next_char();
    ch = peek_char();
  }
}

Parser::Token Parser::next_token() {
  skip_empty();
  UChar32 ch = next_char();
  switch (ch) {
  case EOF: {
    if (is_file_ || use_stream_) {
      return Token::END_OF_FILE;
    }
    return Token::NONE;
  }
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
    UString s;
    s += ch;
    static UErrorCode err;
    static USet set("[()\"'`,;]"_u, err);
    set.freeze();
    read_until(s, set);
    if (s == "1+" || s == "1-") {
      last_symbol_ = new Symbol(s, filename_, row, col);
      return Token::SYMBOL;
    }
    if (u_isdigit(ch) || (s.length() > 1 && (ch == '-' || ch == '+') && u_isdigit(s[1]))) {
      try {
      std::string utf8;
      filename_.toUTF8String(utf8);
        auto num = NumberParser(s, SourceLocation(utf8.c_str(), row)).parse();
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

void Parser::advance(UChar32 curchar) {
  if (curchar == '\n') {
    row_++;
    col_ = 0;
  }
  last_token_ += curchar;
  col_++;
}

// 观览器 #4 的辅助类型
template<class ... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

/**
 * 读取当前字符并前移迭代器，类似于 nextposinc 系列。返回EOF表示读到末尾。
*/
UChar32 Parser::next_char() {
  UChar32 ch = std::visit(overloaded{
      [](UIteratorP iter) -> UChar32 {
        UChar32 ch = iter->next32PostInc();
        return ch == UIteratorDone ? EOF : ch; },
      [](UIterator& iter) -> UChar32 {
        UChar32 ch = iter.next32PostInc();
        return ch == UIteratorDone ? EOF : ch; },
      [](pscm::Port* arg) -> UChar32 { return arg->read_char(); },
    }, code_);
  advance(ch);
  return ch;
}

UChar32 Parser::peek_char() {
  return std::visit(overloaded{
      [](UIteratorP iter) -> UChar32 {
        UChar32 ch = iter->current32();
        return ch == UIteratorDone ? EOF : ch; },
      [](UIterator& iter) -> UChar32 {
        UChar32 ch = iter.current32();
        return ch == UIteratorDone ? EOF : ch; },
      [](pscm::Port* arg) -> UChar32 { return arg->peek_char(); },
    }, code_);
}

void Parser::read_until(UString& s, const USet& end) {
  UChar32 ch = peek_char();
  while (ch != EOF) {
    if (u_isspace(ch)) {
      break;
    }
    if (end.contains(ch)) {
      break;
    }
    s += ch;
    next_char();
    ch = peek_char();
  }
}

Number operator""_num(const char *data, std::size_t len) {
  UString str(data, len);
  NumberParser parser(str);
  return parser.parse();
}
} // namespace pscm
