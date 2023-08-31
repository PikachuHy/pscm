//
// Created by PikachuHy on 2023/3/12.
//

#pragma once
#include <utility>

#include "pscm/Evaluator.h"

namespace pscm {

class Continuation {
public:
  Continuation(Evaluator::Register reg, Evaluator::Stack stack, std::vector<Evaluator::RegisterType> reg_type_stack);
  UString to_string() const;
private:
  Evaluator::Register reg_;
  Evaluator::Stack stack_;
  std::vector<Evaluator::RegisterType> reg_type_stack_;
  friend class Evaluator;
};

} // namespace pscm
