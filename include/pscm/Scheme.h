//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include "pscm/Cell.h"
#include <vector>

namespace pscm {
class SymbolTable;

class Scheme {
public:
  Scheme(bool use_register_machine = false);
  ~Scheme();
  Cell eval(const char *code);
  Cell eval(Cell expr);
  Cell lookup(Cell expr);
  Cell apply(Cell op, Cell args);

private:
  friend Cell scm_define(Scheme& scm, Cell args);
  friend Cell scm_call_proc(Scheme& scm, const Procedure& proc, Cell args);
  std::vector<SymbolTable *> envs_;
  bool use_register_machine_;
};

} // namespace pscm
