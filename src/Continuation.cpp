//
// Created by PikachuHy on 2023/3/12.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Continuation.h"
#include "pscm/misc/ICUCompat.h"
#endif
namespace pscm {
UString Continuation::to_string() const{
  UString out;
  out += "#<continuation ";
  out += "TODO";
  out += " @ ";
  out += pscm::to_string(this);
  out += ">";
  return out;
}

Continuation::Continuation(Evaluator::Register reg, Evaluator::Stack stack,
                           std::vector<Evaluator::RegisterType> reg_type_stack)
    : reg_(std::move(reg))
    , stack_(std::move(stack))
    , reg_type_stack_(std::move(reg_type_stack)) {
}
} // namespace pscm