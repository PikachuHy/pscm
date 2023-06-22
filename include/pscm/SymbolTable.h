//
// Created by PikachuHy on 2023/3/4.
//

#pragma once
#include "pscm/Cell.h"
#include <unordered_map>

namespace pscm {
class Symbol;

class SymbolTable {
public:
  SymbolTable(std::string name = {}, SymbolTable *parent = nullptr)
      : name_(name)
      , parent_(parent) {
  }

  const std::string& name() const {
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

  std::unordered_map<StringView, Entry *> map_;
  std::string name_;
  SymbolTable *parent_;
};

} // namespace pscm
