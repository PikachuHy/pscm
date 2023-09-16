#pragma once
#include "pscm/Cell.h"

namespace pscm {
class Scheme;

class SchemeProxy {
public:
  SchemeProxy(Scheme& scm)
      : scm_(scm) {
  }

  Cell eval(SymbolTable *env, Cell expr);

  Module *current_module() const;

  void set_current_module(Module *m);
  bool load(const UString& filename);

  Module *create_module(Cell module_name);
  bool has_module(Cell module_name) const;
  Module *get_module(Cell module_name) const;
  void load_module(const UString& filename, Cell module_name);
  void vau_hack(Symbol *sym, Cell value);

private:
  Scheme& scm_;
};
} // namespace pscm