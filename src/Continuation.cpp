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
#endif
namespace pscm {
std::ostream& operator<<(std::ostream& out, const Continuation& cont) {
  out << "#<continuation ";
  out << "TODO";
  out << " @ ";
  out << &cont;
  out << ">";
  return out;
}

Continuation::Continuation(Evaluator::Register reg, Evaluator::Stack stack,
                           std::vector<Evaluator::RegisterType> reg_type_stack)
    : reg_(std::move(reg))
    , stack_(std::move(stack))
    , reg_type_stack_(std::move(reg_type_stack)) {
}
} // namespace pscm