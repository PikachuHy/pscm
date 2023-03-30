//
// Created by PikachuHy on 2023/2/23.
//

#include "pscm/Scheme.h"
#include "pscm/Evaluator.h"
#include "pscm/Exception.h"
#include "pscm/Expander.h"
#include "pscm/Function.h"
#include "pscm/Macro.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Parser.h"
#include "pscm/Procedure.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <string>
#include <string_view>
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace pscm {

Cell scm_define(Scheme& scm, Cell args) {
  auto first_arg = car(args);
  if (first_arg.is_sym()) {
    auto sym = first_arg.to_symbol();
    auto ret = scm.eval(cadr(args));
    PSCM_ASSERT(!scm.envs_.empty());
    auto env = scm.envs_.back();
    env->insert(sym, ret);
    return {};
  }
  if (first_arg.is_pair()) {
    auto proc_name = car(first_arg);
    auto proc_args = cdr(first_arg);
    PSCM_ASSERT(proc_name.is_sym());
    auto sym = proc_name.to_symbol();
    auto proc = new Procedure(sym, proc_args, cdr(args), scm.envs_.back());
    PSCM_ASSERT(!scm.envs_.empty());
    auto env = scm.envs_.back();
    env->insert(sym, proc);
    return {};
  }
  throw Exception("Invalid define args: " + args.to_string());
}

Cell scm_lambda(Scheme& scm, Cell args) {
  return new Procedure(nullptr, car(args), cdr(args), scm.cur_env());
}

Cell scm_quote(Scheme& scm, Cell args) {
  return car(args);
}

Cell scm_cond(Scheme& scm, Cell args) {
  if (args.is_nil()) {
    PSCM_THROW_EXCEPTION("Missing clauses (cond)");
  }
  while (!args.is_nil()) {
    auto clause = car(args);
    SPDLOG_INFO("clause: {}", clause);
    args = cdr(args);
    if (clause.is_nil()) {
      PSCM_THROW_EXCEPTION("Bad cond clause " + clause.to_string() + " in expression " + args.to_string());
    }
    auto test = car(clause);
    auto expr = cadr(clause);
    SPDLOG_INFO("test: {}", test);
    SPDLOG_INFO("expr: {}", expr);
    if (test.is_sym()) {
      PSCM_ASSERT(test.to_symbol());
      auto sym = test.to_symbol();
      if (*sym == cond_else) {
        return scm.eval(expr);
      }
    }
    auto ret = scm.eval(test);
    if (ret.is_bool()) {
      if (!ret.to_bool()) {
        continue;
      }
    }
    auto tmp = cdr(clause);
    if (tmp.is_nil()) {
      PSCM_THROW_EXCEPTION("Invalid cond expr: " + clause.to_string());
    }
    SPDLOG_INFO("tmp: {}", tmp);
    auto arrow = car(tmp);
    if (arrow.is_sym() && *arrow.to_symbol() == "=>"_sym) {
      auto recipient = cadr(tmp);
      auto f = scm.eval(recipient);
      auto f_args = list(quote, ret);
      return scm.apply(f, list(f_args));
    }
    else {
      return scm.eval(expr);
    }
  }

  return Cell::none();
}

Cell scm_if(Scheme& scm, Cell args) {
  auto test = car(args);
  auto consequent = cadr(args);
  auto alternate = cddr(args);
  auto pred = scm.eval(test);
  PSCM_ASSERT(pred.is_bool());
  if (pred.to_bool()) {
    return scm.eval(consequent);
  }
  else {
    if (alternate.is_nil()) {
      return Cell::none();
    }
    return scm.eval(car(alternate));
  }
}

Cell scm_and(Scheme& scm, Cell args) {
  Cell pred = Cell::bool_true();
  while (!args.is_nil()) {
    auto expr = car(args);
    args = cdr(args);
    pred = scm.eval(expr);
    if (pred.is_bool() && !pred.to_bool()) {
      return Cell::bool_false();
    }
  }
  return pred;
}

Cell scm_or(Scheme& scm, Cell args) {
  while (!args.is_nil()) {
    auto expr = car(args);
    args = cdr(args);
    auto pred = scm.eval(expr);
    PSCM_ASSERT(pred.is_bool());
    if (pred.to_bool()) {
      return Cell::bool_true();
    }
    else {
      continue;
    }
  }
  return Cell::bool_false();
}

Cell scm_let(Scheme& scm, Cell args) {
  auto expr = expand_let(args);
  return scm.eval(expr);
}

Cell scm_let_star(Scheme& scm, Cell args) {
  auto expr = expand_let_star(args);
  return scm.eval(expr);
}

bool is_member(Cell item, Cell list) {
  while (!list.is_nil()) {
    if (item == car(list)) {
      return true;
    }
    list = cdr(list);
  }
  return false;
}

Cell scm_case(Scheme& scm, Cell args) {
  auto expr = expand_case(args);
  auto ret = scm.eval(expr);
  return ret;
}

Cell lambda = new Macro("lambda", Label::APPLY_LAMBDA, scm_lambda);
Cell quote = new Macro("quote", Label::APPLY_QUOTE, scm_quote);
// TODO: #<primitive-generic for-each>
Cell for_each = new Macro("builtin_for-each", Label::APPLY_FOR_EACH, scm_lambda);
Cell apply = new Macro("builtin_apply", Label::APPLY_APPLY, scm_lambda);

Scheme::Scheme(bool use_register_machine)
    : use_register_machine_(use_register_machine) {
  auto env = new SymbolTable();
  envs_.push_back(env);
  env->insert(new Symbol("+"), new Function("+", add));
  env->insert(new Symbol("-"), new Function("-", minus));
  env->insert(new Symbol("*"), new Function("*", mul));
  env->insert(new Symbol("/"), new Function("/", div));
  env->insert(new Symbol("<"), new Function("<", less_than));
  env->insert(new Symbol("="), new Function("=", equal_to));
  env->insert(new Symbol(">"), new Function(">", greater_than));
  env->insert(new Symbol("negative?"), new Function("negative?", is_negative));
  env->insert(new Symbol("not"), new Function("not", builtin_not));
  env->insert(new Symbol("display"), new Function("display", display));
  env->insert(new Symbol("newline"), new Function("newline", newline));
  env->insert(new Symbol("procedure?"), new Function("procedure?", is_procedure));
  env->insert(new Symbol("boolean?"), new Function("boolean?", is_boolean));
  env->insert(new Symbol("list"), new Function("list", create_list));
  env->insert(new Symbol("list?"), new Function("list?", is_list));
  env->insert(new Symbol("set-cdr!"), new Function("set-cdr!", set_cdr));
  env->insert(new Symbol("assv"), new Function("assv", assv));
  env->insert(new Symbol("cons"), new Function("car", proc_cons));
  env->insert(new Symbol("car"), new Function("car", proc_car));
  env->insert(new Symbol("cdr"), new Function("cdr", proc_cdr));
  env->insert(new Symbol("cadr"), new Function("cadr", proc_cadr));
  env->insert(new Symbol("cdar"), new Function("cdar", proc_cdar));
  env->insert(new Symbol("eqv?"), new Function("eqv?", is_eqv));
  env->insert(new Symbol("eq?"), new Function("eqv?", is_eq));
  env->insert(new Symbol("equal?"), new Function("equal?", is_equal));
  env->insert(new Symbol("memq"), new Function("memq", memq));
  env->insert(new Symbol("memv"), new Function("memv", memv));
  env->insert(new Symbol("member"), new Function("member", member));
  env->insert(new Symbol("make-vector"), new Function("make-vector", make_vector));

  env->insert(new Symbol("define"), new Macro("define", Label::APPLY_DEFINE, scm_define));
  env->insert(new Symbol("cond"), new Macro("cond", Label::APPLY_COND, scm_cond));
  env->insert(new Symbol("if"), new Macro("if", Label::APPLY_IF, scm_if));
  env->insert(new Symbol("and"), new Macro("and", Label::APPLY_AND, scm_and));
  env->insert(new Symbol("or"), new Macro("or", Label::APPLY_OR, scm_or));
  env->insert(new Symbol("set!"), new Macro("set!", Label::APPLY_SET, scm_define));
  env->insert(new Symbol("let"), new Macro("let", Label::APPLY_LET, scm_let));
  env->insert(new Symbol("let*"), new Macro("let*", Label::APPLY_LET_STAR, scm_let_star));
  env->insert(new Symbol("case"), new Macro("case", Label::APPLY_CASE, scm_case));
  env->insert(new Symbol("quote"), quote);
  env->insert(new Symbol("lambda"), lambda);
  {
    auto proc = new Procedure(&callcc, cons(new Symbol("proc"), nil), nil, env);
    env->insert(&callcc, proc);
    env->insert(new Symbol("call/cc"), proc);
  }
  env->insert(new Symbol("for-each"), Procedure::create_for_each(env));
  env->insert(new Symbol("apply"), Procedure::create_apply(env));
  {
    auto proc_args = cons(new Symbol("producer"), cons(new Symbol("consumer"), nil));
    auto proc = new Procedure(&call_with_values, proc_args, nil, env);
    env->insert(&call_with_values, proc);
  }
  {
    auto proc = new Procedure(&values, new Symbol("obj"), nil, env);
    env->insert(&values, proc);
  }
  auto new_env = new SymbolTable(env);
  envs_.push_back(new_env);
}

Scheme::~Scheme() {
  while (!envs_.empty()) {
    auto env = envs_.back();
    delete env;
    envs_.pop_back();
  }
}

Cell Scheme::eval(const char *code) {
  try {
    Parser parser(code);
    auto ret = parser.parse();
    if (ret.is_none()) {
      return Cell::none();
    }
    if (use_register_machine_) {
      ret = Evaluator().eval(ret, envs_.back());
    }
    else {
      ret = eval(ret);
    }
    return ret;
  }
  catch (Exception& ex) {
    SPDLOG_ERROR("eval {} error: {}", code, ex.what());
    return Cell::ex(ex.what());
  }
}

Cell Scheme::eval(Cell expr) {
  if (expr.is_self_evaluated()) {
    return expr;
  }
  if (expr.tag_ == Cell::Tag::SYMBOL) {
    return lookup(expr);
  }
  if (expr.tag_ == Cell::Tag::PAIR) {
    auto op = car(expr);
    op = eval(op);
    SPDLOG_INFO("expr: {}", expr.to_string());
    auto args = cdr(expr);
    SPDLOG_INFO("args: {}", args.to_string());
    return apply(op, args);
  }
  PSCM_THROW_EXCEPTION("unsupported");
  return {};
}

Cell Scheme::lookup(Cell expr) {
  PSCM_ASSERT(expr.is_sym());
  auto sym = expr.to_symbol();
  PSCM_ASSERT(sym);
  PSCM_ASSERT(!envs_.empty());
  auto env = envs_.back();
  auto ret = env->get_or(sym, {});
  if (ret.is_none()) {
    PSCM_THROW_EXCEPTION("Unbound variable: "s + std::string(sym->name()));
  }
  return ret;
}

Cell Scheme::apply(Cell op, Cell args) {
  SPDLOG_INFO("apply op: {}, {}", op.to_string(), args.to_string());
  if (op.is_func()) {
    PSCM_ASSERT(args.is_pair());
    SPDLOG_INFO("apply args: {}", args.to_string());
    // eval each arg
    auto ret = cons(nil, nil);
    auto iter = ret;
    while (!args.is_nil()) {
      auto arg = car(args);
      auto val = eval(arg);
      auto p = cons(val, nil);
      iter->second = p;
      iter = p;
      args = cdr(args);
    }
    auto f = op.to_func();
    SPDLOG_INFO("func {} args: {}", op, ret->second);
    return f->call(ret->second);
  }
  if (op.is_macro()) {
    PSCM_ASSERT(args.is_pair());
    SPDLOG_INFO("apply {} with args: {}", op, args.to_string());
    auto f = op.to_macro();
    return f->call(*this, args);
  }
  if (op.is_proc()) {
    PSCM_ASSERT(args.is_pair() || args.is_nil());
    SPDLOG_INFO("apply args: {}", args.to_string());
    // eval each arg
    auto ret = cons(nil, nil);
    auto iter = ret;
    while (!args.is_nil()) {
      auto arg = car(args);
      auto val = eval(arg);
      auto p = cons(val, nil);
      iter->second = p;
      iter = p;
      args = cdr(args);
    }
    auto proc = op.to_proc();
    auto proc_args = ret->second;
    bool ok = proc->check_args(proc_args);
    if (!ok) {
      PSCM_THROW_EXCEPTION("Wrong number of arguments to " + Cell(proc).to_string());
    }
    auto proc_env = proc->create_proc_env(proc_args);
    envs_.push_back(proc_env);
    auto proc_ret = proc->call(*this, proc_args);
    envs_.pop_back();
    return proc_ret;
  }
  PSCM_THROW_EXCEPTION("unsupported op: " + op.to_string());
  return {};
}

SymbolTable *Scheme::cur_env() const {
  PSCM_ASSERT(!envs_.empty());
  auto env = envs_.back();
  return env;
}
} // namespace pscm