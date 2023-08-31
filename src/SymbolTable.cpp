//
// Created by PikachuHy on 2023/3/4.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/SymbolTable.h"
#include "pscm/Exception.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include "pscm/misc/ICUCompat.h"
#include "unicode/ustream.h"
#include <iostream>
#include <spdlog/fmt/fmt.h>
#include <string>
#endif
using namespace std::string_literals;

namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.SymbolTable");

bool SymbolTable::contains(Symbol *sym) const {
  PSCM_ASSERT(sym);
  if (map_.find(sym->name()) != map_.end()) {
    return true;
  }
  if (parent_) {
    return parent_->contains(sym);
  }
  return false;
}

void SymbolTable::insert(Symbol *sym, Cell cell) {
  PSCM_ASSERT(sym);
  map_[UString(sym->name())] = new Entry{ .data = cell };
}

bool SymbolTable::remove(Symbol *sym) {
  PSCM_ASSERT(sym);
  return map_.erase(sym->name()) > 0;
}

Cell SymbolTable::get(Symbol *sym, SourceLocation loc) const {
  PSCM_ASSERT(sym);
  const auto name = sym->name();
  if (map_.find(name) != map_.end()) {
    return map_.at(name)->data;
  }
  if (parent_) {
    return parent_->get(sym, loc);
  }
  PSCM_THROW_EXCEPTION(loc.to_string() + ", Unbound variable: " + sym->name());
}

Cell SymbolTable::get_or(Symbol *sym, Cell default_value, SourceLocation loc) const {
  PSCM_ASSERT(sym);
  if (map_.find(sym->name()) != map_.end()) {
    return map_.at(sym->name())->data;
  }
  if (parent_) {
    return parent_->get_or(sym, default_value, loc);
  }
  sym->print_debug_info();
  PSCM_THROW_EXCEPTION(loc.to_string() + ", Unbound variable: " + sym->name());
}

void SymbolTable::set(Symbol *sym, Cell value, SourceLocation loc) {
  PSCM_ASSERT(sym);
  auto name = sym->name();
  if (map_.find(name) != map_.end()) {
    map_[name]->data = value;
    return;
  }
  if (parent_) {
    parent_->set(sym, value, loc);
  }
  else {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Unbound variable: " + sym->name());
  }
}

void SymbolTable::use(SymbolTable *env, Symbol *sym) {
  this->map_[sym->name()] = env->map_.at(sym->name());
}

void SymbolTable::use(const SymbolTable& env) {
  for (auto& entry : env.map_) {
    auto sym = entry.first;
    if (this->map_.find(sym) != this->map_.end()) {
      continue;
    }
    this->map_[sym] = entry.second;
  }
}

void SymbolTable::dump(SourceLocation loc) const {
  auto p = this;
  while (p) {
    std::cout << "[" << loc.to_string() << "]";
    std::cout << " ";
    std::cout << (void *)p;
    std::cout << " ";
    std::cout << p->map_.size();
    std::cout << " ";
    std::cout << p->name_;
    std::cout << std::endl;
    p = p->parent_;
  }
}
} // namespace pscm