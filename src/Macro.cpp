//
// Created by PikachuHy on 2023/3/4.
//

#include "pscm/Macro.h"
#include "pscm/common_def.h"

namespace pscm {
Cell Macro::call(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(f_.index() == 1);
  auto f = std::get<1>(f_);
  return (*f)(scm, env, args);
}

Cell Macro::call(Cell args) {
  PSCM_ASSERT(f_.index() == 2);
  auto f = std::get<2>(f_);
  return (*f)(args);
}

std::ostream& operator<<(std::ostream& out, const Macro& macro) {
  out << "#";
  out << "<";
  out << "primitive-built-macro!";
  out << " ";
  out << macro.name_;
  out << ">";
  return out;
}
} // namespace pscm