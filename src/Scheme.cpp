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
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include "pscm/version.h"
#include <string>
#include <string_view>
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace pscm {

Cell scm_define(Scheme& scm, SymbolTable *env, Cell args) {
  auto first_arg = car(args);
  if (first_arg.is_sym()) {
    auto sym = first_arg.to_symbol();
    auto ret = scm.eval(env, cadr(args));
    env->insert(sym, ret);
    return {};
  }
  if (first_arg.is_pair()) {
    auto proc_name = car(first_arg);
    auto proc_args = cdr(first_arg);
    PSCM_ASSERT(proc_name.is_sym());
    auto sym = proc_name.to_symbol();
    auto proc = new Procedure(sym, proc_args, cdr(args), env);
    env->insert(sym, proc);
    return {};
  }
  throw Exception("Invalid define args: " + args.to_string());
}

Cell scm_set(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto k = car(args);
  PSCM_ASSERT(k.is_sym());
  auto sym = k.to_symbol();
  auto v = cadr(args);
  v = scm.eval(env, v);
  bool has_k = env->contains(sym);
  if (!has_k) {
    PSCM_THROW_EXCEPTION("Unbound variable: " + k.to_string());
  }
  env->set(sym, v);
  return Cell::none();
}

Cell scm_lambda(Scheme& scm, SymbolTable *env, Cell args) {
  return new Procedure(nullptr, car(args), cdr(args), env);
}

Cell scm_cond(Scheme& scm, SymbolTable *env, Cell args) {
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
        return expr;
      }
    }
    auto ret = scm.eval(env, test);
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
      auto f = scm.eval(env, recipient);
      auto f_args = list(quote, list(list(ret)));
      return cons(apply, cons(f, f_args));
    }
    else {
      return expr;
    }
  }
  return Cell::none();
}

Cell scm_if(Scheme& scm, SymbolTable *env, Cell args) {
  auto test = car(args);
  auto consequent = cadr(args);
  auto alternate = cddr(args);
  auto pred = scm.eval(env, test);
  PSCM_ASSERT(pred.is_bool());
  if (pred.to_bool()) {
    return consequent;
  }
  else {
    if (alternate.is_nil()) {
      return Cell::none();
    }
    return car(alternate);
  }
}

Cell scm_and(Scheme& scm, SymbolTable *env, Cell args) {
  Cell pred = Cell::bool_true();
  while (args.is_pair() && cdr(args).is_pair()) {
    auto expr = car(args);
    args = cdr(args);
    pred = scm.eval(env, expr);
    if (pred.is_bool() && !pred.to_bool()) {
      return Cell::bool_false();
    }
  }
  if (args.is_nil()) {
    return pred;
  }
  else {
    return car(args);
  }
}

Cell scm_or(Scheme& scm, SymbolTable *env, Cell args) {
  Cell pred = Cell::bool_false();
  while (args.is_pair() && cdr(args).is_pair()) {
    auto expr = car(args);
    args = cdr(args);
    pred = scm.eval(env, expr);
    if (pred.is_bool() && !pred.to_bool()) {
      continue;
    }
    return list(quote, pred);
  }
  if (args.is_nil()) {
    return pred;
  }
  else {
    return car(args);
  }
}

Cell scm_begin(Scheme& scm, SymbolTable *env, Cell args) {
  while (args.is_pair() && cdr(args).is_pair()) {
    [[maybe_unused]] auto ret = scm.eval(env, car(args));
    args = cdr(args);
  }
  if (args.is_nil()) {
    return Cell::none();
  }
  else {
    return car(args);
  }
}

Cell scm_quasiquote(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  QuasiQuotationExpander expander(scm, env);
  auto ret = expander.expand(car(args));
  return ret;
}

Cell scm_map(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  Cell ret;
  auto proc = car(args);
  PSCM_ASSERT(proc.is_sym());
  auto list1 = cadr(args);
  PSCM_ASSERT(list1.is_sym());
  proc = env->get(proc.to_symbol());
  list1 = env->get(list1.to_symbol());
  ret = map(
      [&scm, env, proc](Cell expr, auto loc) {
        return scm.eval(env, cons(proc, list(list(quote, expr))));
      },
      list1);
  return list(quote, ret);
}

Cell lambda = new Macro("lambda", Label::APPLY_LAMBDA);
Cell quote = new Macro("quote", Label::APPLY_QUOTE);
Cell unquote = new Macro("unquote", Label::APPLY_QUOTE);
Cell quasiquote = new Macro("quasiquote", Label::APPLY_QUASIQUOTE, scm_quasiquote);
Cell unquote_splicing = new Macro("unquote-splicing", Label::APPLY_QUOTE, scm_quasiquote);
Cell begin = new Macro("begin", Label::APPLY_BEGIN, scm_begin);
// TODO: #<primitive-generic for-each>
Cell builtin_for_each = new Macro("builtin_for-each", Label::APPLY_FOR_EACH);
Cell builtin_map = new Macro("builtin_map", Label::APPLY_MAP, scm_map);
Cell apply = new Macro("builtin_apply", Label::APPLY_APPLY);

Cell version(Cell) {
  static String ver(std::string() + PSCM_VERSION + " (" + GIT_BRANCH + "@" + GIT_HASH + ")");
  return &ver;
}

Scheme::Scheme(bool use_register_machine)
    : use_register_machine_(use_register_machine) {
  auto env = new SymbolTable();
  envs_.push_back(env);
  env->insert(new Symbol("version"), new Function("version", version));
  env->insert(new Symbol("+"), new Function("+", add));
  env->insert(new Symbol("-"), new Function("-", minus));
  env->insert(new Symbol("*"), new Function("*", mul));
  env->insert(new Symbol("/"), new Function("/", div));
  env->insert(new Symbol("<"), new Function("<", less_than));
  env->insert(new Symbol("<="), new Function("<=", less_or_equal_than));
  env->insert(new Symbol("="), new Function("=", equal_to));
  env->insert(new Symbol(">"), new Function(">", greater_than));
  env->insert(new Symbol(">="), new Function(">=", greater_or_equal_than));
  env->insert(new Symbol("negative?"), new Function("negative?", is_negative));
  env->insert(new Symbol("not"), new Function("not", builtin_not));
  env->insert(new Symbol("display"), new Function("display", display));
  env->insert(new Symbol("newline"), new Function("newline", newline));
  env->insert(new Symbol("procedure?"), new Function("procedure?", is_procedure));
  env->insert(new Symbol("boolean?"), new Function("boolean?", is_boolean));
  env->insert(new Symbol("list"), new Function("list", create_list));
  env->insert(new Symbol("list?"), new Function("list?", is_list));
  env->insert(new Symbol("pair?"), new Function("pair?", is_pair));
  env->insert(new Symbol("set-cdr!"), new Function("set-cdr!", set_cdr));
  env->insert(new Symbol("set-car!"), new Function("set-car!", set_car));
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
  env->insert(new Symbol("assq"), new Function("assq", assq));
  env->insert(new Symbol("assv"), new Function("assv", assv));
  env->insert(new Symbol("assoc"), new Function("assoc", assoc));
  env->insert(new Symbol("make-vector"), new Function("make-vector", make_vector));
  env->insert(new Symbol("vector"), new Function("vector", proc_vector));
  env->insert(new Symbol("vector-set!"), new Function("vector-set!", vector_set));
  env->insert(new Symbol("zero?"), new Function("zero?", is_zero));
  env->insert(new Symbol("null?"), new Function("null?", is_null));
  env->insert(new Symbol("length"), new Function("length", length));
  env->insert(new Symbol("append"), new Function("append", append));
  env->insert(new Symbol("reverse"), new Function("reverse", reverse));
  env->insert(new Symbol("list-ref"), new Function("list-ref", list_ref));
  env->insert(new Symbol("expt"), new Function("expt", expt));
  env->insert(new Symbol("abs"), new Function("abs", proc_abs));
  env->insert(new Symbol("sqrt"), new Function("sqrt", proc_sqrt));
  env->insert(new Symbol("round"), new Function("round", proc_round));
  env->insert(new Symbol("inexact->exact"), new Function("inexact->exact", inexact_to_exact));
  env->insert(new Symbol("symbol?"), new Function("symbol?", is_symbol));
  env->insert(new Symbol("symbol->string"), new Function("symbol->string", symbol_to_string));
  env->insert(new Symbol("string->symbol"), new Function("string->symbol", string_to_symbol));
  env->insert(new Symbol("string=?"), new Function("string=?", is_string_equal));

  env->insert(new Symbol("define"), new Macro("define", Label::APPLY_DEFINE, scm_define));
  env->insert(new Symbol("cond"), new Macro("cond", Label::APPLY_COND, scm_cond));
  env->insert(new Symbol("if"), new Macro("if", Label::APPLY_IF, scm_if));
  env->insert(new Symbol("and"), new Macro("and", Label::APPLY_AND, scm_and));
  env->insert(new Symbol("or"), new Macro("or", Label::APPLY_OR, scm_or));
  env->insert(new Symbol("set!"), new Macro("set!", Label::APPLY_SET, scm_set));
  env->insert(new Symbol("let"), new Macro("let", expand_let));
  env->insert(new Symbol("let*"), new Macro("let*", expand_let_star));
  env->insert(new Symbol("letrec"), new Macro("letrec", expand_letrec));
  env->insert(new Symbol("case"), new Macro("case", expand_case));
  env->insert(new Symbol("do"), new Macro("do", expand_do));
  env->insert(new Symbol("quote"), quote);
  env->insert(new Symbol("unquote"), unquote);
  env->insert(new Symbol("unquote-splicing"), unquote_splicing);
  env->insert(new Symbol("lambda"), lambda);
  env->insert(new Symbol("begin"), begin);
  env->insert(new Symbol("quasiquote"), quasiquote);
  {
    auto proc = new Procedure(&callcc, cons(new Symbol("proc"), nil), nil, env);
    env->insert(&callcc, proc);
    env->insert(new Symbol("call/cc"), proc);
  }
  env->insert(new Symbol("for-each"), Procedure::create_for_each(env));
  env->insert(new Symbol("map"), Procedure::create_map(env));
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
      ret = Evaluator(*this).eval(ret, envs_.back());
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

Cell Scheme::eval_args(pscm::SymbolTable *env, pscm::Cell args, SourceLocation loc) {
  auto ret = map(
      [this, env](auto expr, auto) {
        return this->eval(env, expr);
      },
      args, loc);
  return ret;
}

Cell Scheme::eval(pscm::SymbolTable *env, pscm::Cell expr) {
  Cell proc;
  Cell args;
  while (true) {
    SPDLOG_INFO("eval: {}", expr);
    if (expr.is_none()) {
      return expr;
    }
    if (expr.is_self_evaluated()) {
      return expr;
    }
    if (expr.is_sym()) {
      return lookup(env, expr);
    }
    proc = eval(env, car(expr));
    args = cdr(expr);
    if (proc.is_func()) {
      SPDLOG_INFO("proc args: {}", args);
      PSCM_ASSERT(args.is_pair() || args.is_nil());
      auto f = proc.to_func();
      auto func_args = eval_args(env, args);
      return f->call(func_args);
    }
    else if (proc.is_macro()) {
      PSCM_ASSERT(args.is_pair() || args.is_nil());
      auto f = proc.to_macro();
      if (f == quote) {
        return car(args);
      }
      else if (f == apply) {
        auto op = car(args);
        auto op_args = cdr(args);
        SPDLOG_INFO("op: {}", op);
        SPDLOG_INFO("args: {}", op_args);
        op = eval(env, op);
        op_args = eval(env, op_args);
        SPDLOG_INFO("op: {}", op);
        SPDLOG_INFO("args: {}", op_args);
        // TODO:
        if (op.is_func()) {
          return op.to_func()->call(car(op_args));
        }
        PSCM_ASSERT("unsupported now");
      }
      else if (f == lambda) {
        return new Procedure(nullptr, car(args), cdr(args), env);
      }
      if (f->is_func()) {
        expr = f->call(args);
      }
      else {
        expr = f->call(*this, env, args);
      }
      continue;
    }
    else if (proc.is_proc()) {
      PSCM_ASSERT(args.is_pair() || args.is_nil());
      auto f = proc.to_proc();
      auto proc_args = eval_args(env, args);
      bool ok = f->check_args(proc_args);
      if (!ok) {
        PSCM_THROW_EXCEPTION("Wrong number of arguments to " + Cell(proc).to_string());
      }
      env = f->create_proc_env(proc_args);
      auto body = f->body();
      while (body.is_pair() && cdr(body).is_pair()) {
        [[maybe_unused]] auto ret = eval(env, car(body));
        body = cdr(body);
      }
      expr = car(body);
      continue;
    }
    else {
      SPDLOG_ERROR("unsupported {}", proc);
      PSCM_THROW_EXCEPTION("unsupported");
    }
  }
}

Cell Scheme::eval(Cell expr) {
  PSCM_ASSERT(!envs_.empty());
  auto env = envs_.back();
  auto ret = eval(env, expr);
  return ret;
}

Cell Scheme::lookup(SymbolTable *env, Cell expr, SourceLocation loc) {
  PSCM_ASSERT(env);
  PSCM_ASSERT(expr.is_sym());
  auto sym = expr.to_symbol();
  PSCM_ASSERT(sym);
  auto ret = env->get_or(sym, {}, loc);
  if (ret.is_none()) {
    PSCM_THROW_EXCEPTION("Unbound variable: "s + std::string(sym->name()));
  }
  return ret;
}

} // namespace pscm
