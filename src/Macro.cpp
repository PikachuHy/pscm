//
// Created by PikachuHy on 2023/3/4.
//

#include "pscm/Macro.h"
#include "pscm/ApiManager.h"
#include "pscm/Module.h"
#include "pscm/Procedure.h"
#include "pscm/Scheme.h"
#include "pscm/SchemeProxy.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"

namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Macro");

Cell Macro::call(Scheme& scm, SymbolTable *env, Cell args) {
  if (f_.index() == 1) {
    auto f = std::get<1>(f_);
    return (*f)(scm, env, args);
  }
  if (f_.index() == 4) {
    auto f = std::get<4>(f_);
    return (*f)(scm, env, args);
  }
  else if (f_.index() == 3) {
    auto proc = std::get<3>(f_);
    auto ret = scm.call_proc(env, proc, args);
    ret = scm.eval(env, ret);
    PSCM_DEBUG("expand result: {}", ret);
    return ret;
  }
  else {
    PSCM_THROW_EXCEPTION("not supported now, macro index: " + std::to_string(f_.index()));
  }
}

Cell Macro::call(Cell args) {
  PSCM_ASSERT(f_.index() == 2);
  auto f = std::get<2>(f_);
  return (*f)(args);
}

std::ostream& operator<<(std::ostream& out, const Macro& macro) {
  out << "#";
  out << "<";
  if (macro.is_proc()) {
    out << "macro!";
  }
  else {
    out << "primitive-builtin-macro!";
  }

  out << " ";
  out << macro.name_;
  out << ">";
  return out;
}

Symbol *scm_define_macro(SchemeProxy scm, SymbolTable *env, Cell args) {
  auto first_arg = car(args);
  Procedure *proc;
  Symbol *sym;
  if (first_arg.is_sym()) {
    sym = first_arg.to_sym();
    auto ret = scm.eval(env, cadr(args));
    PSCM_ASSERT(ret.is_proc());
    proc = ret.to_proc();
  }
  else {
    auto proc_name = car(first_arg);
    auto proc_args = cdr(first_arg);
    PSCM_INFO("{} {}", proc_name, proc_args);
    PSCM_ASSERT(proc_name.is_sym());
    sym = proc_name.to_sym();
    proc = new Procedure(sym, proc_args, cdr(args), env);
  }
  env->insert(sym, new Macro(std::string(sym->name()), proc));
  return sym;
}

PSCM_DEFINE_BUILTIN_MACRO(Macro, "define-macro", Label::APPLY_DEFINE_MACRO) {
  scm_define_macro(scm, env, args);
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_MACRO(Macro, "define-public-macro", Label::APPLY_DEFINE_MACRO) {
  auto sym = scm_define_macro(scm, env, args);
  scm.current_module()->export_symbol(sym);
  // FIXME: module system
  scm.current_module()->env()->insert(sym, env->get(sym));
  scm.vau_hack(sym, env->get(sym));
  return Cell::none();
}
} // namespace pscm