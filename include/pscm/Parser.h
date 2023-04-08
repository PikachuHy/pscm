//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include "pscm/Cell.h"

namespace pscm {
class Number;

class Parser {
public:
  Parser(std::string code)
      : code_(std::move(code)){};
  Cell parse();

private:
  Cell parse_expr();
  Cell parse_literal();
  Cell parse_string();
  void skip_empty();
  void skip_line();
  void eat(char ch);
  enum class Token {
    NONE,              //
    LEFT_PARENTHESES,  // (
    RIGHT_PARENTHESES, // )
    NUMBER,            // 0123456789
    SHARP,             // #
    QUOTE,             // '
    UNQUOTE,           // ,
    UNQUOTE_SPLICING,  // ,@
    DOT,               // .
    SEMICOLON,         // ;
    QUOTATION,         // "
    QUASIQUOTE,        // `
    BACK_SLASH,        // forward slash '/', and back slash '\'
    SYMBOL             //
  };
  Token next_token();
  Cell parse_token(Token token, std::size_t start);

private:
  std::string code_;
  std::size_t pos_{};
  bool has_parsed_ = false;
  Number *last_num_ = nullptr;
  Symbol *last_symbol_ = nullptr;
  bool has_dot = false;
};

} // namespace pscm
