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
  Macro(std::string name, Label pos, Cell::ScmMacro f)
      : name_(std::move(name))
      , pos_(pos)
      , f_(f) {
  }

  Cell call(Scheme& scm, Cell args);

  Label pos() const {
    return pos_;
  }

  friend std::ostream& operator<<(std::ostream& out, const Macro& macro);

private:
  std::string name_;
  Label pos_;
  std::variant<std::monostate, Cell::ScmMacro> f_;
};

} // namespace pscm
