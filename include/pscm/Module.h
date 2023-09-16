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

  Cell name() const {
    return name_;
  }

  SymbolTable *env() const {
    return env_;
  }

  Cell export_sym_list();

  void export_symbol(Symbol *sym);
  void use_module(Module *m, bool use_all = false);
  UString to_string() const;

private:
  Cell name_;
  SymbolTable *env_;
  std::unordered_set<Cell> export_sym_list_;
};

Cell module_map(SchemeProxy scm, SymbolTable *env, Cell args);
} // namespace pscm