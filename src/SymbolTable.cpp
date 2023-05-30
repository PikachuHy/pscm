//
// Created by PikachuHy on 2023/3/4.
//

#include "pscm/SymbolTable.h"
#include "pscm/Exception.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include <string>
using namespace std::string_literals;

namespace pscm {
bool SymbolTable::contains(Symbol *sym) const {
  PSCM_ASSERT(sym);
  if (map_.contains(sym->name())) {
    return true;
  }
  if (parent_) {
    return parent_->contains(sym);
  }
  return false;
}

void SymbolTable::insert(Symbol *sym, Cell cell) {
  PSCM_ASSERT(sym);
  map_[sym->name()] = new Entry{ .data = cell };
}

bool SymbolTable::remove(Symbol *sym) {
  PSCM_ASSERT(sym);
  return map_.erase(sym->name()) > 0;
}

Cell SymbolTable::get(Symbol *sym, SourceLocation loc) const {
  PSCM_ASSERT(sym);
  auto name = sym->name();
  if (map_.contains(name)) {
    return map_.at(name)->data;
  }
  if (parent_) {
    return parent_->get(sym, loc);
  }
  PSCM_THROW_EXCEPTION(loc.to_string() + ", Unbound variable: "s + std::string(sym->name()));
}

Cell SymbolTable::get_or(Symbol *sym, Cell default_value, SourceLocation loc) const {
  PSCM_ASSERT(sym);
  if (map_.contains(sym->name())) {
    return map_.at(sym->name())->data;
  }
  if (parent_) {
    return parent_->get_or(sym, default_value, loc);
  }
  sym->print_debug_info();
  PSCM_THROW_EXCEPTION(loc.to_string() + ", Unbound variable: "s + std::string(sym->name()));
}

void SymbolTable::set(Symbol *sym, Cell value, SourceLocation loc) {
  PSCM_ASSERT(sym);
  auto name = sym->name();
  if (map_.contains(name)) {
    map_[name]->data = value;
    return;
  }
  if (parent_) {
    parent_->set(sym, value, loc);
  }
  else {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Unbound variable: "s + std::string(sym->name()));
  }
}

void SymbolTable::use(SymbolTable *env, Symbol *sym) {
  this->map_[sym->name()] = env->map_.at(sym->name());
}

void SymbolTable::use(const SymbolTable& env) {
  for (auto [sym, val] : env.map_) {
    if (this->map_.contains(sym)) {
      continue;
    }
    this->map_[sym] = val;
  }
}
} // namespace pscm