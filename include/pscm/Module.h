#pragma once
#include "pscm/Cell.h"
#include <string>
#include <unordered_set>

namespace pscm {
class SymbolTable;

class Module {
public:
  Module(Cell name, SymbolTable *env)
      : name_(name)
      , env_(env) {
  }

  SymbolTable *env() const {
    return env_;
  }

  Cell export_sym_list();

  void export_symbol(Symbol *sym);
  void use_module(Module *m);
  friend std::ostream& operator<<(std::ostream& os, const Module& m);

private:
  Cell name_;
  SymbolTable *env_;
  std::unordered_set<Cell> export_sym_list_;
};

Cell module_map(SchemeProxy scm, SymbolTable *env, Cell args);
} // namespace pscm