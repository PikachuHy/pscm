//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Cell.h"

namespace pscm {
class Scheme;
class Evaluator;
class SymbolTable;

class Procedure {
public:
  Procedure(Symbol *name, Cell args, Cell body, SymbolTable *env)
      : name_(name)
      , args_(std::move(args))
      , body_(std::move(body))
      , env_(env) {
  }

  friend std::ostream& operator<<(std::ostream& out, const Procedure& proc);

  Cell args() const {
    return args_;
  }

  Cell call(Scheme& scm, Cell args) const;

  bool check_args(Cell args) const;

  SymbolTable *create_proc_env(Cell args) const;

  static Procedure *create_for_each(SymbolTable *env);

  static Procedure *create_apply(SymbolTable *env);

private:
  Symbol *name_;
  Cell args_;
  Cell body_;
  SymbolTable *env_;
  friend class Evaluator;
};

} // namespace pscm
