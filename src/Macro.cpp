//
// Created by PikachuHy on 2023/3/4.
//

#include "pscm/Macro.h"
#include "pscm/Procedure.h"
#include "pscm/Scheme.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"

namespace pscm {
Cell Macro::call(Scheme& scm, SymbolTable *env, Cell args) {
  if (f_.index() == 1) {
    auto f = std::get<1>(f_);
    return (*f)(scm, env, args);
  }
  else if (f_.index() == 3) {
    auto proc = std::get<3>(f_);
    auto ret = scm.call_proc(env, proc, args);
    ret = scm.eval(env, ret);
    SPDLOG_INFO("expand result: {}", ret);
    return ret;
  }
  else {
    PSCM_THROW_EXCEPTION("not supported now, macro index: " + std::to_string(f_.index()));
  }
}

Cell Macro::call(Cell args) {
  PSCM_ASSERT(f_.index() == 2);
  auto f = std::get<2>(f_);
  return (*f)(args);
}

std::ostream& operator<<(std::ostream& out, const Macro& macro) {
  out << "#";
  out << "<";
  if (macro.is_func()) {
    out << "primitive-built-macro!";
  }
  else {
    out << "macro!";
  }

  out << " ";
  out << macro.name_;
  out << ">";
  return out;
}
} // namespace pscm