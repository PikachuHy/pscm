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
#include "pscm/Promise.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include "pscm/version.h"
#include "spdlog/spdlog.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
using namespace std::string_literals;
using namespace std::string_view_literals;
namespace fs = std::filesystem;

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

Cell scm_define_macro(Scheme& scm, SymbolTable *env, Cell args) {
  auto first_arg = car(args);
  PSCM_ASSERT(first_arg.is_pair());
  auto proc_name = car(first_arg);
  auto proc_args = cdr(first_arg);
  PSCM_ASSERT(proc_name.is_sym());
  auto sym = proc_name.to_symbol();
  auto proc = new Procedure(sym, proc_args, cdr(args), env);
  env->insert(sym, new Macro(std::string(sym->name()), proc));
  return {};
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
  auto args_bak = args;
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
    SPDLOG_INFO("test: {}", test);
    auto expr = cdr(clause);
    SPDLOG_INFO("expr: {}", expr);
    if (test.is_sym()) {
      PSCM_ASSERT(test.to_symbol());
      auto sym = test.to_symbol();
      if (*sym == cond_else) {
        if (expr.is_nil()) {
          PSCM_THROW_EXCEPTION("Bad cond clause (else) in expression " + args_bak.to_string());
        }
        return car(expr);
      }
    }
    auto ret = scm.eval(env, test);
    if (ret.is_bool()) {
      if (!ret.to_bool()) {
        continue;
      }
    }
    if (expr.is_nil()) {
      return Cell::bool_true();
    }
    auto tmp = cdr(clause);
    SPDLOG_INFO("tmp: {}", tmp);
    auto arrow = car(tmp);
    auto arrow_sym = "=>"_sym;
    if (arrow.is_sym() && *arrow.to_symbol() == arrow_sym && !env->contains(&arrow_sym)) {
      auto recipient = cadr(tmp);
      auto f = scm.eval(env, recipient);
      auto f_args = list(quote, list(list(ret)));
      return cons(apply, cons(f, f_args));
    }
    else {
      while (expr.is_pair() && cdr(expr).is_pair()) {
        [[maybe_unused]] auto ret = scm.eval(env, car(expr));
        expr = cdr(expr);
      }
      return car(expr);
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
  auto lists = cdr(args);
  SPDLOG_INFO("lists: {}", lists);
  PSCM_ASSERT(lists.is_sym());
  proc = env->get(proc.to_symbol());
  lists = env->get(lists.to_symbol());
  int len = 0;
  ret = for_each(
      [&len](auto, auto) {
        len++;
      },
      lists);
  switch (len) {
  case 0: {
    break;
  }
  case 1: {
    ret = map(
        [&scm, env, proc](Cell expr, auto loc) {
          return scm.eval(env, cons(proc, list(list(quote, expr))));
        },
        car(lists));
    break;
  }
  case 2: {
    ret = map(
        [&scm, env, proc](Cell expr1, Cell expr2, auto loc) {
          return scm.eval(env, cons(proc, list(list(quote, expr1), list(quote, expr2))));
        },
        car(lists), cadr(lists));
    break;
  }
  default: {
    PSCM_THROW_EXCEPTION("not supported now");
  }
  }
  return list(quote, ret);
}

Cell scm_for_each(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  Cell ret;
  auto proc = car(args);
  PSCM_ASSERT(proc.is_sym());
  auto lists = cdr(args);
  PSCM_ASSERT(lists.is_sym());
  proc = env->get(proc.to_symbol());
  lists = env->get(lists.to_symbol());
  int len = 0;
  ret = for_each(
      [&len](auto, auto) {
        len++;
      },
      lists);
  switch (len) {
  case 0: {
    break;
  }
  case 1: {
    ret = for_each(
        [&scm, env, proc](Cell expr, auto loc) {
          [[maybe_unused]] auto ret = scm.eval(env, cons(proc, list(list(quote, expr))));
        },
        car(lists));
    break;
  }
  case 2: {
    ret = for_each(
        [&scm, env, proc](Cell expr1, Cell expr2, auto loc) {
          [[maybe_unused]] auto ret = scm.eval(env, cons(proc, list(list(quote, expr1), list(quote, expr2))));
        },
        car(lists), cadr(lists));
    break;
  }
  default: {
    PSCM_THROW_EXCEPTION("not supported now");
  }
  }
  return list(quote, ret);
}

Cell scm_delay(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  Cell expr = car(args);
  auto proc = new Procedure(nullptr, nil, list(expr), env);
  return new Promise(proc);
}

Cell scm_force(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  Cell promise = car(args);
  PSCM_ASSERT(promise.is_sym());
  // SPDLOG_INFO("promise: {}", promise);
  // promise = scm.eval(env, promise);
  promise = env->get(promise.to_symbol());
  PSCM_ASSERT(promise.is_promise());
  auto p = promise.to_promise();
  Cell ret;
  if (p->ready()) {
    ret = p->result();
  }
  else {
    auto proc = p->proc();
    ret = scm.eval(env, list(proc));
    p->set_result(ret);
  }
  return list(quote, ret);
}

Cell debug_set(Scheme& scm, SymbolTable *env, Cell args) {
  return Cell::none();
}

Cell scm_load(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_sym());
  auto sym = arg.to_symbol();
  auto val = env->get(sym);
  PSCM_ASSERT(val.is_str());
  auto s = val.to_str();
  auto filename = s->str();
  bool ok = scm.load(std::string(filename).c_str());
  return Cell(ok);
}

Cell scm_eval(Scheme& scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  PSCM_ASSERT(arg.is_sym());
  auto sym = arg.to_symbol();
  auto expr = env->get(sym);
  auto ret = scm.eval(env, expr);
  return ret;
}

Cell lambda = new Macro("lambda", Label::APPLY_LAMBDA);
Cell quote = new Macro("quote", Label::APPLY_QUOTE);
Cell unquote = new Macro("unquote", Label::APPLY_QUOTE);
Cell quasiquote = new Macro("quasiquote", Label::APPLY_QUASIQUOTE, scm_quasiquote);
Cell unquote_splicing = new Macro("unquote-splicing", Label::APPLY_QUOTE, scm_quasiquote);
Cell begin = new Macro("begin", Label::APPLY_BEGIN, scm_begin);
// TODO: #<primitive-generic for-each>
Cell builtin_for_each = new Macro("builtin_for-each", Label::APPLY_FOR_EACH, scm_for_each);
Cell builtin_map = new Macro("builtin_map", Label::APPLY_MAP, scm_map);
Cell builtin_force = new Macro("builtin_force", Label::APPLY_FORCE, scm_force);
Cell builtin_load = new Macro("builtin_load", Label::APPLY_LOAD, scm_load);
Cell builtin_eval = new Macro("builtin_eval", Label::APPLY_EVAL, scm_eval);
Cell apply = new Macro("builtin_apply", Label::APPLY_APPLY);

Cell version(Cell) {
  static String ver(std::string() + PSCM_VERSION + " (" + GIT_BRANCH + "@" + GIT_HASH + ")");
  return &ver;
}

void Scheme::add_func(Symbol *sym, Function *func) {
  PSCM_ASSERT(sym);
  PSCM_ASSERT(func);
  envs_.back()->insert(sym, func);
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
  env->insert(new Symbol("positive?"), new Function("positive?", is_positive));
  env->insert(new Symbol("negative?"), new Function("negative?", is_negative));
  env->insert(new Symbol("odd?"), new Function("odd?", is_odd));
  env->insert(new Symbol("even?"), new Function("even?", is_even));
  env->insert(new Symbol("max"), new Function("max", proc_max));
  env->insert(new Symbol("min"), new Function("min", proc_min));
  env->insert(new Symbol("quotient"), new Function("quotient", quotient));
  env->insert(new Symbol("remainder"), new Function("remainder", remainder));
  env->insert(new Symbol("modulo"), new Function("modulo", modulo));
  env->insert(new Symbol("gcd"), new Function("gcd", proc_gcd));
  env->insert(new Symbol("lcm"), new Function("lcm", proc_lcm));
  env->insert(new Symbol("not"), new Function("not", builtin_not));
  env->insert(new Symbol("display"), new Function("display", display));
  env->insert(new Symbol("write"), new Function("write", write));
  env->insert(new Symbol("newline"), new Function("newline", newline));
  env->insert(new Symbol("procedure?"), new Function("procedure?", is_procedure));
  env->insert(new Symbol("boolean?"), new Function("boolean?", is_boolean));
  env->insert(new Symbol("list"), new Function("list", create_list));
  env->insert(new Symbol("list?"), new Function("list?", is_list));
  env->insert(new Symbol("pair?"), new Function("pair?", is_pair));
  env->insert(new Symbol("set-cdr!"), new Function("set-cdr!", set_cdr));
  env->insert(new Symbol("set-car!"), new Function("set-car!", set_car));
  env->insert(new Symbol("assv"), new Function("assv", assv));
  env->insert(new Symbol("cons"), new Function("cons", proc_cons));
  env->insert(new Symbol("car"), new Function("car", proc_car));
  env->insert(new Symbol("caar"), new Function("caar", proc_caar));
  env->insert(new Symbol("cdr"), new Function("cdr", proc_cdr));
  env->insert(new Symbol("cadr"), new Function("cadr", proc_cadr));
  env->insert(new Symbol("cdar"), new Function("cdar", proc_cdar));
  env->insert(new Symbol("cddr"), new Function("cddr", proc_cddr));
  env->insert(new Symbol("caddr"), new Function("cddr", proc_caddr));
  env->insert(new Symbol("eqv?"), new Function("eqv?", is_eqv));
  env->insert(new Symbol("eq?"), new Function("eqv?", is_eq));
  env->insert(new Symbol("equal?"), new Function("equal?", is_equal));
  env->insert(new Symbol("memq"), new Function("memq", memq));
  env->insert(new Symbol("memv"), new Function("memv", memv));
  env->insert(new Symbol("member"), new Function("member", member));
  env->insert(new Symbol("assq"), new Function("assq", assq));
  env->insert(new Symbol("assv"), new Function("assv", assv));
  env->insert(new Symbol("assoc"), new Function("assoc", assoc));
  env->insert(new Symbol("vector?"), new Function("vector?", is_vector));
  env->insert(new Symbol("make-vector"), new Function("make-vector", make_vector));
  env->insert(new Symbol("vector"), new Function("vector", proc_vector));
  env->insert(new Symbol("vector-length"), new Function("vector-length", vector_length));
  env->insert(new Symbol("vector-ref"), new Function("vector-ref", vector_ref));
  env->insert(new Symbol("vector-set!"), new Function("vector-set!", vector_set));
  env->insert(new Symbol("vector->list"), new Function("vector->list", vector_to_list));
  env->insert(new Symbol("list->vector"), new Function("list->vector", list_to_vector));
  env->insert(new Symbol("vector-fill!"), new Function("vector-fill!", vector_fill));
  env->insert(new Symbol("zero?"), new Function("zero?", is_zero));
  env->insert(new Symbol("null?"), new Function("null?", is_null));
  env->insert(new Symbol("length"), new Function("length", length));
  env->insert(new Symbol("append"), new Function("append", append));
  env->insert(new Symbol("reverse"), new Function("reverse", reverse));
  env->insert(new Symbol("list-ref"), new Function("list-ref", list_ref));
  env->insert(new Symbol("acos"), new Function("acos", proc_acos));
  env->insert(new Symbol("expt"), new Function("expt", expt));
  env->insert(new Symbol("abs"), new Function("abs", proc_abs));
  env->insert(new Symbol("sqrt"), new Function("sqrt", proc_sqrt));
  env->insert(new Symbol("round"), new Function("round", proc_round));
  env->insert(new Symbol("exact?"), new Function("exact?", is_exact));
  env->insert(new Symbol("inexact?"), new Function("inexact?", is_inexact));
  env->insert(new Symbol("inexact->exact"), new Function("inexact->exact", inexact_to_exact));
  env->insert(new Symbol("symbol?"), new Function("symbol?", is_symbol));
  env->insert(new Symbol("symbol->string"), new Function("symbol->string", symbol_to_string));
  env->insert(new Symbol("string->symbol"), new Function("string->symbol", string_to_symbol));
  env->insert(new Symbol("string?"), new Function("string?", is_string));
  env->insert(new Symbol("make-string"), new Function("make-string", make_string));
  env->insert(new Symbol("string"), new Function("string", proc_string));
  env->insert(new Symbol("string-length"), new Function("string-length", string_length));
  env->insert(new Symbol("string-ref"), new Function("string-ref", string_ref));
  env->insert(new Symbol("string-set!"), new Function("string-set!", string_set));
  env->insert(new Symbol("string=?"), new Function("string=?", is_string_equal));
  env->insert(new Symbol("string-ci=?"), new Function("string-ci=?", is_string_equal_case_insensitive));
  env->insert(new Symbol("string<?"), new Function("string<?", is_string_less));
  env->insert(new Symbol("string>?"), new Function("string>?", is_string_greater));
  env->insert(new Symbol("string<=?"), new Function("string<=?", is_string_less_or_equal));
  env->insert(new Symbol("string>=?"), new Function("string>=?", is_string_greater_or_equal));
  env->insert(new Symbol("string-ci<?"), new Function("string-ci<?", is_string_less_case_insensitive));
  env->insert(new Symbol("string-ci>?"), new Function("string-ci>?", is_string_greater_case_insensitive));
  env->insert(new Symbol("string-ci<=?"), new Function("string-ci<=?", is_string_less_or_equal_case_insensitive));
  env->insert(new Symbol("string-ci>=?"), new Function("string-ci>=?", is_string_greater_or_equal_case_insensitive));
  env->insert(new Symbol("substring"), new Function("substring", proc_substring));
  env->insert(new Symbol("string-append"), new Function("string-append", string_append));
  env->insert(new Symbol("string->list"), new Function("string->list", string_to_list));
  env->insert(new Symbol("list->string"), new Function("list->string", list_to_string));
  env->insert(new Symbol("string-copy"), new Function("string-copy", string_copy));
  env->insert(new Symbol("string-fill"), new Function("string-fill", string_fill));
  env->insert(new Symbol("number?"), new Function("number?", is_number));
  env->insert(new Symbol("complex?"), new Function("complex?", is_complex));
  env->insert(new Symbol("real?"), new Function("real?", is_real));
  env->insert(new Symbol("integer?"), new Function("integer?", is_integer));
  env->insert(new Symbol("rational?"), new Function("rational?", is_rational));
  env->insert(new Symbol("string->number"), new Function("string->number", string_to_number));
  env->insert(new Symbol("number->string"), new Function("number->string", number_to_string));
  env->insert(new Symbol("char?"), new Function("char?", is_char));
  env->insert(new Symbol("char=?"), new Function("char=?", is_char_equal));
  env->insert(new Symbol("char<?"), new Function("char<?", is_char_less));
  env->insert(new Symbol("char>?"), new Function("char>?", is_char_greater));
  env->insert(new Symbol("char<=?"), new Function("char<=?", is_char_less_or_equal));
  env->insert(new Symbol("char>=?"), new Function("char>=?", is_char_greater_or_equal));
  env->insert(new Symbol("char-ci=?"), new Function("char-ci=?", is_char_equal_case_insensitive));
  env->insert(new Symbol("char-ci<?"), new Function("char-ci<?", is_char_less_case_insensitive));
  env->insert(new Symbol("char-ci>?"), new Function("char-ci>?", is_char_greater_case_insensitive));
  env->insert(new Symbol("char-ci<=?"), new Function("char-ci<=?", is_char_less_or_equal_case_insensitive));
  env->insert(new Symbol("char-ci>=?"), new Function("char-ci>=?", is_char_greater_or_equal_case_insensitive));
  env->insert(new Symbol("char-alphabetic?"), new Function("char-alphabetic?", is_char_alphabetic));
  env->insert(new Symbol("char-numeric?"), new Function("char-numeric?", is_char_numeric));
  env->insert(new Symbol("char-whitespace?"), new Function("char-whitespace?", is_char_whitespace));
  env->insert(new Symbol("char-upper-case?"), new Function("char-upper-case?", is_char_upper_case));
  env->insert(new Symbol("char-lower-case?"), new Function("char-lower-case?", is_char_lower_case));
  env->insert(new Symbol("integer->char"), new Function("integer->char", integer_to_char));
  env->insert(new Symbol("char->integer"), new Function("char->integer", char_to_integer));
  env->insert(new Symbol("char-upcase"), new Function("char-upcase", char_upcase));
  env->insert(new Symbol("char-downcase"), new Function("char-downcase", char_downcase));
  env->insert(new Symbol("input-port?"), new Function("input-port?", is_input_port));
  env->insert(new Symbol("output-port?"), new Function("output-port?", is_output_port));
  env->insert(new Symbol("current-input-port"), new Function("current-input-port", current_input_port));
  env->insert(new Symbol("current-output-port"), new Function("current-output-port", current_output_port));
  env->insert(new Symbol("open-input-file"), new Function("open-input-file", open_input_file));
  env->insert(new Symbol("open-output-file"), new Function("open-output-file", open_output_file));
  env->insert(new Symbol("close-input-port"), new Function("close-input-port", close_input_port));
  env->insert(new Symbol("close-output-port"), new Function("close-output-port", close_output_port));
  env->insert(new Symbol("read"), new Function("read", proc_read));
  env->insert(new Symbol("read-char"), new Function("read-char", read_char));
  env->insert(new Symbol("peek-char"), new Function("peek-char", peek_char));
  env->insert(new Symbol("write-char"), new Function("write-char", write_char));
  env->insert(new Symbol("eof-object?"), new Function("eof-object", is_eof_object));
  env->insert(new Symbol("char-ready?"), new Function("char-ready?", is_char_ready));
  env->insert(new Symbol("transcript-on"), new Function("transcript-on", transcript_on));
  env->insert(new Symbol("transcript-off"), new Function("transcript-off", transcript_off));

  env->insert(new Symbol("exit"), new Function("exit", proc_exit));

  env->insert(new Symbol("define"), new Macro("define", Label::APPLY_DEFINE, scm_define));
  env->insert(new Symbol("cond"), new Macro("cond", Label::APPLY_COND, scm_cond));
  env->insert(new Symbol("if"), new Macro("if", Label::APPLY_IF, scm_if));
  env->insert(new Symbol("and"), new Macro("and", Label::APPLY_AND, scm_and));
  env->insert(new Symbol("or"), new Macro("or", Label::APPLY_OR, scm_or));
  env->insert(new Symbol("set!"), new Macro("set!", Label::APPLY_SET, scm_set));
  env->insert(new Symbol("delay"), new Macro("delay", Label::APPLY_DELAY, scm_delay));
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
  env->insert(new Symbol("force"), Procedure::create_force(env));
  env->insert(new Symbol("load"), Procedure::create_load(env));
  env->insert(new Symbol("eval"), Procedure::create_eval(env));
  env->insert(new Symbol("call-with-output-string"), Procedure::create_call_with_output_string(env));
  {
    auto proc_args = cons(new Symbol("producer"), cons(new Symbol("consumer"), nil));
    auto proc = new Procedure(&call_with_values, proc_args, nil, env);
    env->insert(&call_with_values, proc);
  }
  {
    auto proc = new Procedure(&values, new Symbol("obj"), nil, env);
    env->insert(&values, proc);
  }
  // TODO: texmacs support
  env->insert(new Symbol("debug-set!"), new Macro("debug-set!", Label::APPLY_DEFINE, debug_set));
  env->insert(new Symbol("define-public"), new Macro("define-public", Label::APPLY_DEFINE, scm_define));
  env->insert(new Symbol("primitive-load"), new Macro("primitive-load", Label::APPLY_DEFINE, scm_define));
  env->insert(new Symbol("current-module"), new Function("current-module", current_module));
  env->insert(new Symbol("define-macro"), new Macro("define-macro", Label::APPLY_DEFINE_MACRO, scm_define_macro));

  env->insert(new Symbol("gensym"), new Function("gensym", proc_gensym));

  eval(R"(
(define (call-with-output-file filename proc)
  (let* ((port (open-output-file filename))
        (ret (apply proc (list port))))
    (close-output-port port)
    ret))
)");
  eval(R"(
(define (call-with-input-file filename proc)
  (let* ((port (open-input-file filename))
         (ret (apply proc (list port))))
    (close-input-port port)
    ret))
)");
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

bool Scheme::load(const char *filename) {
  std::cout << "load: " << filename << std::endl;
  if (!fs::exists(filename)) {
    SPDLOG_ERROR("file not found: {}", filename);
    return false;
  }
  std::fstream ifs;
  ifs.open(filename, std::ios::in);
  if (!ifs.is_open()) {
    SPDLOG_ERROR("load file {} error", filename);
    return false;
  }
  ifs.seekg(0, ifs.end);
  auto sz = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::string code;
  code.resize(sz);
  ifs.read(code.data(), sz);
  try {
    Parser parser(code, filename);
    Cell expr = parser.next();
    while (!expr.is_none()) {
      if (use_register_machine_) {
        Evaluator(*this).eval(expr, envs_.back());
      }
      else {
        eval(expr);
      }
      expr = parser.next();
    }
  }
  catch (Exception& ex) {
    SPDLOG_ERROR("load file {} error", filename);
    return false;
  }
  return true;
}

Cell Scheme::eval_args(pscm::SymbolTable *env, pscm::Cell args, SourceLocation loc) {
  auto ret = map(
      [this, env](auto expr, auto) {
        return this->eval(env, expr);
      },
      args, loc);
  return ret;
}

extern Cell construct_apply_argl(Cell argl);

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
      SPDLOG_INFO("proc {} args: {}", proc, args.pretty_string());
      PSCM_ASSERT(args.is_pair() || args.is_nil());
      auto f = proc.to_func();
      auto func_args = eval_args(env, args);
      auto ret = f->call(func_args);
      SPDLOG_INFO("proc {} ret: {}", expr.pretty_string(), ret.pretty_string());
      return ret;
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
        op = eval(env, op);
        op_args = eval(env, op_args);
        SPDLOG_INFO("op: {}", op);
        SPDLOG_INFO("args: {}", op_args);
        args = construct_apply_argl(op_args);
        // TODO:
        if (op.is_func()) {
          return op.to_func()->call(args);
        }
        else if (op.is_proc()) {
          expr = call_proc(env, op.to_proc(), args);
          // expr = cons(op, car(op_args));
          SPDLOG_INFO("new expr: {}", expr);
          continue;
        }
        PSCM_THROW_EXCEPTION("unsupported now: " + op.to_string());
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
      expr = call_proc(env, f, proc_args);
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
  bool has_sym = env->contains(sym);
  if (!has_sym) {
    sym->print_debug_info();
    PSCM_THROW_EXCEPTION("Unbound variable: "s + std::string(sym->name()));
  }
  auto ret = env->get_or(sym, {}, loc);
  return ret;
}

Cell Scheme::call_proc(SymbolTable *& env, Procedure *proc, Cell args, SourceLocation loc) {
  bool ok = proc->check_args(args);
  if (!ok) {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Wrong number of arguments to " + Cell(proc).to_string());
  }
  env = proc->create_proc_env(args);
  auto body = proc->body();
  SPDLOG_INFO("body: {}", body);
  while (body.is_pair() && cdr(body).is_pair()) {
    [[maybe_unused]] auto ret = eval(env, car(body));
    body = cdr(body);
  }
  if (body.is_nil()) {
    return nil;
  }
  else {
    return car(body);
  }
}
} // namespace pscm
