//
// Created by PikachuHy on 2023/3/12.
//

#include "pscm/Continuation.h"

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