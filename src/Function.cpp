//
// Created by PikachuHy on 2023/3/4.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Function.h"
#include "pscm/ApiManager.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <ostream>
#endif
namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Function");
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

UString Function::to_string() const{
  UString out;
  out += "#<primitive-generic "
    + name_
    + ">";
  return out;
}

PSCM_DEFINE_BUILTIN_PROC(Function, "noop") {
  return Cell::bool_false();
}
} // namespace pscm