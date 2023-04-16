//
// Created by PikachuHy on 2023/2/25.
//

#pragma once
#include <string>
#include <string_view>

namespace pscm {

class Symbol {
public:
  Symbol(std::string_view sym)
      : name_(sym.data(), sym.size()) {
  }

  Symbol(std::string_view sym, std::string_view filename, std::size_t row, std::size_t col)
      : name_(sym.data(), sym.size())
      , filename_(filename)
      , row_(row)
      , col_(col) {
  }

  std::string_view name() const {
    return name_;
  }

  friend std::ostream& operator<<(std::ostream& out, const Symbol& sym);
  bool operator==(const Symbol& sym) const;

  void print_debug_info();

private:
  std::string name_;
  std::string_view filename_;
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
