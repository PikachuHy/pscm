//
// Created by PikachuHy on 2023/2/23.
//

#pragma once
#include "pscm/Cell.h"
#include <unordered_map>
#include <vector>

namespace pscm {
class SymbolTable;
class SchemeProxy;

class Scheme {
public:
  Scheme(bool use_register_machine = false);
  ~Scheme();
  Cell eval(const char *code);
  void eval_all(const char *code, SourceLocation loc = {});
  Cell eval(Cell expr);
  bool load(const char *filename);
  void add_func(Symbol *sym, Function *func);
  void repl();

private:
  [[nodiscard]] Cell eval(SymbolTable *env, Cell expr);
  Cell eval_internal(SymbolTable *env, const char *code);
  [[nodiscard]] Cell eval_args(SymbolTable *env, Cell args, SourceLocation loc = {});
  [[nodiscard]] Cell lookup(SymbolTable *env, Cell expr, SourceLocation loc = {});
  [[nodiscard]] Cell call_proc(SymbolTable *& env, Procedure *proc, Cell args, SourceLocation loc = {});
  Module *create_module(Cell name);

  Module *current_module() const {
    return current_module_;
  };

  void load_module(const std::string& filename, Cell module_name);
  friend Cell debug_set(Scheme& scm, SymbolTable *env, Cell args);
  friend class QuasiQuotationExpander;
  friend class Macro;
  std::vector<SymbolTable *> envs_;
  SymbolTable *root_env_;
  SymbolTable *root_derived_env_;
  Module *current_module_;
  std::vector<Module *> module_list_;
  std::unordered_map<Cell, Module *> module_map_;
  bool use_register_machine_;
  bool in_repl_ = false;
  friend class SchemeProxy;
};

} // namespace pscm
