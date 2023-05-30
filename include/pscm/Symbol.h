//
// Created by PikachuHy on 2023/2/25.
//

#pragma once
#include "pscm/Cell.h"
#include <string>
#include <string_view>

namespace pscm {

class Symbol {
public:
  Symbol(std::string name)
      : name_(std::move(name)) {
  }

  Symbol(std::string name, std::string_view filename, std::size_t row, std::size_t col)
      : name_(std::move(name))
      , filename_(filename)
      , row_(row)
      , col_(col) {
  }

  std::string_view name() const {
    return name_;
  }

  friend std::ostream& operator<<(std::ostream& out, const Symbol& sym);
  bool operator==(const Symbol& sym) const;

  HashCodeType hash_code() const;
  void print_debug_info();
  static Symbol for_each;
  static Symbol map;
  static Symbol load;
  static Symbol quasiquote;
  static Symbol unquote_splicing;

private:
  std::string name_;
  std::string filename_;
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
