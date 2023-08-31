//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Cell.h"
#include "pscm/misc/ICUCompat.h"
#include <unordered_map>

namespace pscm {
class Symbol;

class SymbolTable {
public:
  SymbolTable(UString name = {}, SymbolTable *parent = nullptr)
      : name_(name)
      , parent_(parent) {
  }

  const UString& name() const {
    return name_;
  }

  bool contains(Symbol *sym) const;
  void insert(Symbol *sym, Cell cell);
  bool remove(Symbol *sym);
  Cell get(Symbol *sym, SourceLocation loc = {}) const;
  Cell get_or(Symbol *sym, Cell default_value, SourceLocation loc = {}) const;
  void set(Symbol *sym, Cell value, SourceLocation loc = {});

  SymbolTable *parent() const {
    return parent_;
  }

  void use(SymbolTable *env, Symbol *sym);
  void use(const SymbolTable& env);

  void dump(SourceLocation loc = {}) const;

private:
  struct Entry {
    Cell data;
  };

  std::unordered_map<UString, Entry *> map_;
  UString name_;
  SymbolTable *parent_;
};

} // namespace pscm
