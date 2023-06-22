//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include "pscm/Cell.h"
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace pscm {
class Number;

class Parser {
public:
  Parser(std::string code);
  Parser(std::string code, StringView filename);
  Parser(std::istream *in);
  Cell parse();
  Cell next();

private:
  Cell parse_expr();
  Cell parse_literal();
  Cell parse_string();
  void skip_empty();
  void skip_line();
  void eat(char ch);
  bool is_eof() const;
  enum class Token {
    NONE,              //
    END_OF_FILE,       // end of file
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
  void advance();
  char next_char();
  char peek_char();
  void read_until(std::string& s, StringView end);

private:
  std::string code_;
  std::size_t pos_{};
  std::istream *in_ = nullptr;
  bool use_stream_ = false;
  bool has_parsed_ = false;
  Number *last_num_ = nullptr;
  Symbol *last_symbol_ = nullptr;
  bool has_dot = false;
  bool is_file_ = false;
  std::size_t row_ = 0;
  std::size_t col_ = 0;
  std::vector<StringView> lines_;
  StringView filename_;
};

} // namespace pscm
