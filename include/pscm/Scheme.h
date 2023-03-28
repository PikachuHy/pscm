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
  [[nodiscard]] Cell eval(const char *code);
  [[nodiscard]] Cell eval(Cell expr);
  [[nodiscard]] Cell lookup(Cell expr);
  [[nodiscard]] Cell apply(Cell op, Cell args);
  [[nodiscard]] SymbolTable *cur_env() const;

private:
  friend Cell scm_define(Scheme& scm, Cell args);
  std::vector<SymbolTable *> envs_;
  bool use_register_machine_;
};

} // namespace pscm
