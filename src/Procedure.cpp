//
// Created by PikachuHy on 2023/3/4.
//

#include "pscm/Procedure.h"
#include "pscm/Exception.h"
#include "pscm/Pair.h"
#include "pscm/Scheme.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"

namespace pscm {
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

Cell scm_call_proc(Scheme& scm, const Procedure& proc, Cell args) {
  PSCM_ASSERT(!scm.envs_.empty());
  auto env = scm.envs_.back();
  auto proc_env = new SymbolTable(env);
  auto params = proc.args();
  PSCM_ASSERT(params.is_pair());
  PSCM_ASSERT(args.is_pair());
  auto p1 = params;
  auto p2 = args;
  while (!p1.is_nil() && !car(p1).is_nil()) {
    if (car(p2).is_nil()) {
      PSCM_THROW_EXCEPTION("Wrong number of arguments to " + Cell(&proc).to_string());
    }
    p1 = cdr(p1);
    p2 = cdr(p2);
  }
  if (!p2.is_nil()) {
    PSCM_THROW_EXCEPTION("Wrong number of arguments to " + Cell(&proc).to_string());
  }
  p1 = params;
  p2 = args;

  while (!p1.is_nil() && !car(p1).is_nil()) {
    PSCM_ASSERT(car(p1).is_sym());
    auto sym = car(p1).to_symbol();
    auto ret = scm.eval(car(p2));
    proc_env->insert(sym, ret);
    p1 = cdr(p1);
    p2 = cdr(p2);
  }
  scm.envs_.push_back(proc_env);
  auto ret = proc.call(scm, args);
  scm.envs_.pop_back();
  delete proc_env;
  return ret;
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
  auto proc_env = new SymbolTable(env_);
  auto p1 = args_;
  auto p2 = args;
  while (p1.is_pair() && !p1.is_nil() && !car(p1).is_nil()) {
    PSCM_ASSERT(car(p1).is_sym());
    auto sym = car(p1).to_symbol();
    auto ret = car(p2);
    proc_env->insert(sym, ret);
    p1 = cdr(p1);
    p2 = cdr(p2);
  }
  if (p1.is_sym()) {
    proc_env->insert(p1.to_symbol(), p2);
  }
  return proc_env;
}

Procedure *Procedure::create_for_each(SymbolTable *env) {
  auto name = new Symbol("for-each");
  auto proc = new Symbol("proc");
  auto list1 = new Symbol("list1");
  Cell args = cons(proc, cons(list1, nil));
  Cell body = cons(for_each, cons(proc, cons(list1, nil)));
  body = cons(body, nil);
  return new Procedure(name, args, body, env);
}

Procedure *Procedure::create_apply(SymbolTable *env) {
  auto name = new Symbol("apply");
  auto proc = new Symbol("proc");
  auto args = new Symbol("args");
  Cell body = cons(apply, cons(proc, cons(args, nil)));
  body = cons(body, nil);
  return new Procedure(name, cons(proc, args), body, env);
}
} // namespace pscm