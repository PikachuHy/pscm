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
  void load(const char *filename);

private:
  [[nodiscard]] Cell eval(SymbolTable *env, Cell expr);
  [[nodiscard]] Cell eval_args(SymbolTable *env, Cell args, SourceLocation loc = {});
  [[nodiscard]] Cell lookup(SymbolTable *env, Cell expr, SourceLocation loc = {});
  [[nodiscard]] Cell call_proc(SymbolTable *& env, Procedure *proc, Cell args, SourceLocation loc = {});
  friend Cell scm_define(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_set(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_cond(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_if(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_and(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_or(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_begin(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_quasiquote(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_map(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_for_each(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_delay(Scheme& scm, SymbolTable *env, Cell args);
  friend Cell scm_force(Scheme& scm, SymbolTable *env, Cell args);
  friend class QuasiQuotationExpander;
  std::vector<SymbolTable *> envs_;
  bool use_register_machine_;
};

} // namespace pscm
