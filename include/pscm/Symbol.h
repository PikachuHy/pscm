//
// Created by PikachuHy on 2023/2/25.
//

#pragma once
#include "pscm/Cell.h"
#include "pscm/misc/ICUCompat.h"
#include <string>

namespace pscm {

class Symbol {
public:
  Symbol(UString name)
      : name_(std::move(name)) {
  }

  Symbol(UString name, const UString & filename, std::size_t row, std::size_t col)
      : name_(std::move(name))
      , filename_(filename)
      , row_(row)
      , col_(col) {
  }

  const UString name() const {
    return get_const_string(name_);
  }

  bool operator==(const Symbol& sym) const;

  HashCodeType hash_code() const;
  void print_debug_info();
  UString to_string() const;
  static Symbol for_each;
  static Symbol map;
  static Symbol load;
  static Symbol quasiquote;
  static Symbol unquote_splicing;

private:
  UString name_;
  UString filename_;
  std::size_t row_;
  std::size_t col_;
};

Symbol operator""_sym(const char *data, std::size_t len);
extern Symbol callcc;
extern Symbol call_with_values;
extern Symbol values;
extern Symbol cond_else;
extern Symbol sym_if;
Symbol *gensym();
} // namespace pscm
