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

private:
  [[nodiscard]] Cell eval(SymbolTable *env, Cell expr);
  [[nodiscard]] Cell eval_args(SymbolTable *env, Cell args);
  [[nodiscard]] Cell lookup(SymbolTable *env, Cell expr, SourceLocation loc = {});
  friend Cell scm_define(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_set(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_cond(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_if(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_and(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_or(Scheme& scm, SymbolTable *env, Cell args);
  std::vector<SymbolTable *> envs_;
  bool use_register_machine_;
};

} // namespace pscm
