//
// Created by PikachuHy on 2023/3/4.
//

#include "pscm/Function.h"
#include "pscm/ApiManager.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <ostream>

namespace pscm {
Cell Function::call(Cell args, SourceLocation loc) {
  PSCM_ASSERT(f_.index() == 1 || f_.index() == 2);
  if (f_.index() == 1) {
    auto f = std::get<1>(f_);
    return (*f)(args);
  }
  else if (f_.index() == 2) {
    auto f = std::get<2>(f_);
    return (*f)(args, loc);
  }
  else {
    PSCM_THROW_EXCEPTION("Invalid function");
  }
}

std::ostream& operator<<(std::ostream& out, const Function& func) {
  out << "#";
  out << "<";
  out << "primitive-generic";
  out << " ";
  out << func.name_;
  out << ">";
  return out;
}

PSCM_DEFINE_BUILTIN_PROC(Function, "noop") {
  return Cell::bool_false();
}
} // namespace pscm