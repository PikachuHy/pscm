//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include "pscm/Cell.h"
#include "unicode/chariter.h"
#include "unicode/schriter.h"
#include <iosfwd>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace pscm {
class Number;

class Parser {
public:
  Parser(const UString & code);
  Parser(UIteratorP in);
  Parser(const UString & code, const UString& filename);
  Parser(Port* in);
  Cell parse();
  Cell next();

private:
  Cell parse_expr();
  Cell parse_literal();
  Cell parse_string();
  void skip_empty();
  void skip_line();
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
  Cell parse_token(Token token);
  void advance(UChar32 curchar);
  UChar32 next_char();
  UChar32 peek_char();
  void read_until(UString& s, const USet& end);

private:
  std::variant<UIterator, UIteratorP, Port*> code_;
  UString last_token_;
  bool use_stream_ = false;
  bool has_parsed_ = false;
  Number *last_num_ = nullptr;
  Symbol *last_symbol_ = nullptr;
  bool has_dot = false;
  bool is_file_ = false;
  std::size_t row_ = 0;
  std::size_t col_ = 0;
  std::vector<UString> lines_;
  const UString filename_;
};

} // namespace pscm
