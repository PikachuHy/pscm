//
// Created by PikachuHy on 2023/3/4.
//

#include "pscm/Procedure.h"
#include "pscm/ApiManager.h"
#include "pscm/Exception.h"
#include "pscm/Expander.h"
#include "pscm/Pair.h"
#include "pscm/Scheme.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"

namespace pscm {
PSCM_INLINE_LOG_DECLARE("pscm.core.Procedure");
std::ostream& operator<<(std::ostream& out, const Procedure& proc) {
  out << "#";
  out << "<procedure ";
  if (proc.name_) {
    out << proc.name_->name();
  }
  else {
    out << "#f";
  }
  out << " ";
  out << proc.args_;
  out << ">";
  return out;
}

Cell Procedure::call(Scheme& scm, Cell args) const {
  auto code = body_;
  Cell ret{};
  while (!code.is_nil()) {
    ret = scm.eval(car(code));
    code = cdr(code);
  }
  return ret;
}

bool Procedure::check_args(Cell args) const {
  auto p1 = args_;
  auto p2 = args;
  while (!p1.is_nil() && p1.is_pair() && !car(p1).is_nil()) {
    if (p2.is_nil()) {
      return false;
    }
    p1 = cdr(p1);
    p2 = cdr(p2);
  }
  if (!p2.is_nil()) {
    return p1.is_sym();
  }
  return true;
}

SymbolTable *Procedure::create_proc_env(Cell args) const {
  auto proc_env = new SymbolTable("apply proc", env_);
  auto p1 = args_;
  auto p2 = args;
  while (p1.is_pair() && !p1.is_nil() && !car(p1).is_nil()) {
    PSCM_ASSERT(car(p1).is_sym());
    auto sym = car(p1).to_sym();
    auto ret = car(p2);
    proc_env->insert(sym, ret);
    p1 = cdr(p1);
    p2 = cdr(p2);
  }
  if (p1.is_sym()) {
    proc_env->insert(p1.to_sym(), p2);
  }
  return proc_env;
}

Procedure *Procedure::create_apply(SymbolTable *env) {
  auto name = new Symbol("apply");
  auto proc = new Symbol("proc");
  auto args = new Symbol("args");
  Cell body = cons(apply, cons(proc, args));
  body = list(body);
  return new Procedure(name, cons(proc, args), body, env);
}

PSCM_DEFINE_BUILTIN_PROC(Procedure, "procedure-name") {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_proc());
  auto proc = arg.to_proc();
  return proc->name();
}
} // namespace pscm