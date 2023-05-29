//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Cell.h"
#include <string>
#include <variant>

namespace pscm {
class Scheme;
class Procedure;

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

  Macro(std::string name, Label pos, Cell::ScmMacro2 f)
      : name_(std::move(name))
      , pos_(pos)
      , f_(f) {
  }

  Macro(std::string name, Cell::ScmFunc f)
      : name_(std::move(name))
      , pos_(Label::APPLY_MACRO)
      , f_(f) {
  }

  Macro(std::string name, Procedure *proc)
      : name_(std::move(name))
      , pos_(Label::APPLY_MACRO)
      , f_(proc) {
  }

  [[nodiscard]] Cell call(Scheme& scm, SymbolTable *env, Cell args);
  [[nodiscard]] Cell call(Cell args);

  [[nodiscard]] Label pos() const {
    return pos_;
  }

  bool is_func() const {
    return f_.index() == 2;
  }

  bool is_proc() const {
    return f_.index() == 3;
  }

  Procedure *to_proc() const {
    return std::get<3>(f_);
  }

  std::string name() const {
    return name_;
  }

  friend std::ostream& operator<<(std::ostream& out, const Macro& macro);

private:
  std::string name_;
  Label pos_;
  std::variant<std::monostate, Cell::ScmMacro, Cell::ScmFunc, Procedure *, Cell::ScmMacro2> f_;
};

} // namespace pscm
