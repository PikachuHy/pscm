//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Cell.h"
#include <string>
#include <variant>

namespace pscm {
class Scheme;

class Macro {
public:
  Macro(std::string name, Label pos)
      : name_(std::move(name))
      , pos_(pos) {
  }

  Macro(std::string name, Label pos, Cell::ScmMacro f)
      : name_(std::move(name))
      , pos_(pos)
      , f_(f) {
  }

  Macro(std::string name, Cell::ScmFunc f)
      : name_(std::move(name))
      , pos_(Label::APPLY_MACRO)
      , f_(f) {
  }

  [[nodiscard]] Cell call(Scheme& scm, SymbolTable *env, Cell args);
  [[nodiscard]] Cell call(Cell args);

  [[nodiscard]] Label pos() const {
    return pos_;
  }

  bool is_func() const {
    return f_.index() == 2;
  }

  friend std::ostream& operator<<(std::ostream& out, const Macro& macro);

private:
  std::string name_;
  Label pos_;
  std::variant<std::monostate, Cell::ScmMacro, Cell::ScmFunc> f_;
};

} // namespace pscm
