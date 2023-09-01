//
// Created by PikachuHy on 2023/2/23.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Evaluator.h"
#include "pscm/common_def.h"

import pscm;
import std;
import fmt;
import linenoise;
#else
#include "pscm/ApiManager.h"
#include "pscm/Displayable.h"
#include "pscm/Evaluator.h"
#include "pscm/Exception.h"
#include "pscm/Expander.h"
#include "pscm/Function.h"
#include "pscm/HashTable.h"
#include "pscm/Logger.h"
#include "pscm/Macro.h"
#include "pscm/Module.h"
#include "pscm/Number.h"
#include "pscm/Pair.h"
#include "pscm/Parser.h"
#include "pscm/Procedure.h"
#include "pscm/Promise.h"
#include "pscm/Scheme.h"
#include "pscm/SchemeProxy.h"
#include "pscm/Str.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/logger/Appender.h"
#include "pscm/misc/ICUCompat.h"
#include "pscm/scm_utils.h"
#include "pscm/version.h"
#include "spdlog/spdlog.h"
#include "unicode/schriter.h"
#include <fstream>
#include <linenoise.hpp>
#include <string>
#include <string_view>
#endif
using namespace std::string_literals;

PSCM_INLINE_LOG_DECLARE("pscm.core.Scheme");

namespace pscm {

Symbol *scm_define(SchemeProxy scm, SymbolTable *env, Cell args) {
  PSCM_DEBUG("args: {0}", args);
  PSCM_ASSERT(args.is_pair());
  auto key = car(args);
  auto val = cdr(args);
  while (!key.is_sym()) {
    auto proc_name = car(key);
    auto proc_args = cdr(key);
    Cell proc = cons(lambda, cons(proc_args, val));
    key = proc_name;
    val = list(proc);
  }
  PSCM_ASSERT(key.is_sym());
  auto sym = key.to_sym();
  val = scm.eval(env, car(val));
  if (val.is_proc() && (cadr(args).is_pair() || car(args).is_pair())) {
    auto proc = val.to_proc();
    PSCM_ASSERT(proc);
    proc->set_name(sym);
  }
  env->insert(sym, val);
  return sym;
}

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "define", Label::APPLY_DEFINE) {
  scm_define(scm, env, args);
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "define-public", Label::APPLY_DEFINE) {
  auto sym = scm_define(scm, env, args);
  scm.current_module()->export_symbol(sym);
  // FIXME: wrong module or wrong env?
  scm.current_module()->env()->insert(sym, env->get(sym));
  scm.vau_hack(sym, env->get(sym));
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "set!", Label::APPLY_SET) {
  PSCM_ASSERT(args.is_pair());
  auto k = car(args);
  PSCM_ASSERT(k.is_sym());
  auto sym = k.to_sym();
  auto v = cadr(args);
  v = scm.eval(env, v);
  bool has_k = env->contains(sym);
  if (!has_k) {
    PSCM_ERROR("Unbound variable: {0} from {1}", k, k.source_location());
    PSCM_THROW_EXCEPTION("Unbound variable: " + k.to_string());
  }
  env->set(sym, v);
  return Cell::none();
}

Cell scm_lambda(Scheme& scm, SymbolTable *env, Cell args) {
  return new Procedure(nullptr, car(args), cdr(args), env);
}

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "cond", Label::APPLY_COND) {
  auto args_bak = args;
  if (args.is_nil()) {
    PSCM_THROW_EXCEPTION("Missing clauses (cond)");
  }
  while (!args.is_nil()) {
    auto clause = car(args);
    PSCM_DEBUG("clause: {0}", clause);
    args = cdr(args);
    if (clause.is_nil()) {
      PSCM_THROW_EXCEPTION("Bad cond clause " + clause.to_string() + " in expression " + args.to_string());
    }
    auto test = car(clause);
    PSCM_DEBUG("test: {0}", test);
    auto expr = cdr(clause);
    PSCM_DEBUG("expr: {0}", expr);
    if (test.is_sym()) {
      PSCM_ASSERT(test.to_sym());
      auto sym = test.to_sym();
      if (*sym == cond_else) {
        if (expr.is_nil()) {
          PSCM_THROW_EXCEPTION("Bad cond clause (else) in expression " + args_bak.to_string());
        }
        while (expr.is_pair() && cdr(expr).is_pair()) {
          [[maybe_unused]] auto ret = scm.eval(env, car(expr));
          expr = cdr(expr);
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
    PSCM_DEBUG("tmp: {0}", tmp);
    auto arrow = car(tmp);
    auto arrow_sym = "=>"_sym;
    if (arrow.is_sym() && *arrow.to_sym() == arrow_sym && !env->contains(&arrow_sym)) {
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

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "if", Label::APPLY_IF) {
  auto test = car(args);
  auto consequent = cadr(args);
  auto alternate = cddr(args);
  auto pred = scm.eval(env, test);
  if (!pred.is_bool() || pred.to_bool()) {
    return consequent;
  }
  else {
    if (alternate.is_nil()) {
      return Cell::none();
    }
    return car(alternate);
  }
}

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "and", Label::APPLY_AND) {
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

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "or", Label::APPLY_OR) {
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

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "begin", Label::APPLY_BEGIN) {
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

Cell scm_quasiquote(SchemeProxy scm, SymbolTable *env, Cell args) {
  PSCM_ASSERT(args.is_pair());
  QuasiQuotationExpander expander(scm, env);
  auto ret = expander.expand(car(args));
  return ret;
}

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "quasiquote", Label::APPLY_QUASIQUOTE) {
  return scm_quasiquote(scm, env, args);
}

Cell debug_set(Scheme& scm, SymbolTable *env, Cell args) {
  return Cell::none();
}

Cell lambda = new Macro("lambda", Label::APPLY_LAMBDA);
Cell quote = new Macro("quote", Label::APPLY_QUOTE);
Cell unquote = new Macro("unquote", Label::APPLY_QUOTE);
Cell apply = new Macro("builtin_apply", Label::APPLY_APPLY);

Cell version(Cell) {
  static String ver(UString(VersionInfo::PSCM_VERSION) + " (" +
                    UString(VersionInfo::GIT_BRANCH) + "@" + UString(VersionInfo::GIT_HASH) + ")");
  return &ver;
}

void Scheme::add_func(Symbol *sym, Function *func) {
  PSCM_ASSERT(sym);
  PSCM_ASSERT(func);
  root_derived_env_->insert(sym, func);
}

Scheme::Scheme(bool use_register_machine)
    : use_register_machine_(use_register_machine) {
  // add default ConsoleAppender
  if (pscm::logger::Logger::root_logger()->appender_set().empty()) {
    pscm::logger::Logger::root_logger()->add_appender(new pscm::logger::ConsoleAppender());
  }
  current_module_ = nullptr;
  auto env = new SymbolTable("root");
  // hack: force load code in AList.cpp
  AList a;
  ApiManager::install_api(env);
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

  env->insert(new Symbol("let"), new Macro("let", expand_let));
  env->insert(new Symbol("let*"), new Macro("let*", expand_let_star));
  env->insert(new Symbol("letrec"), new Macro("letrec", expand_letrec));
  env->insert(new Symbol("case"), new Macro("case", expand_case));
  env->insert(new Symbol("do"), new Macro("do", expand_do));
  env->insert(new Symbol("quote"), quote);
  env->insert(new Symbol("unquote"), unquote);
  env->insert(new Symbol("lambda"), lambda);
  {
    auto proc = new Procedure(&callcc, cons(new Symbol("proc"), nil), nil, env);
    env->insert(&callcc, proc);
    env->insert(new Symbol("call/cc"), proc);
  }
  env->insert(new Symbol("apply"), Procedure::create_apply(env));
  env->insert(new Symbol("call-with-output-string"), Procedure::create_call_with_output_string(env));
  env->insert(new Symbol("call-with-input-string"), Procedure::create_call_with_input_string(env));
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
  env->insert(new Symbol("%load-path"), nil);

  eval_internal(env, R"(
(define (call-with-output-file filename proc)
  (let* ((port (open-output-file filename))
        (ret (apply proc (list port))))
    (close-output-port port)
    ret))
)"_u);
  eval_internal(env, R"(
(define (call-with-input-file filename proc)
  (let* ((port (open-input-file filename))
         (ret (apply proc (list port))))
    (close-input-port port)
    ret))
)"_u);
  root_env_ = env;
  env->insert(new Symbol("map-in-order"), env->get(&Symbol::map));
  env->insert(new Symbol("primitive-load"), env->get(&Symbol::load));
  vau_hack_env_ = new SymbolTable("vau hack", env);
  root_derived_env_ = new SymbolTable("root-derived", vau_hack_env_);
  envs_.push_back(root_derived_env_);
  current_module_ = create_module(nil);
  module_list_.push_back(current_module_);
}

Scheme::~Scheme() {
  while (!envs_.empty()) {
    auto env = envs_.back();
    delete env;
    envs_.pop_back();
  }
}

Cell Scheme::eval(const UString& code) {
  try {
    Parser parser(code);
    auto ret = parser.parse();
    if (ret.is_none()) {
      return Cell::none();
    }
    if (in_repl_ && ret.is_sym()) {
      if (!current_module_->env()->contains(ret.to_sym())) {
        std::cout << "ERROR: Unbound variable: " << ret.to_string() << std::endl;
        std::cout << "ABORT: (unbound-variable)" << std::endl;
        return Cell::none();
      }
    }
    if (use_register_machine_) {
      ret = Evaluator(*this).eval(ret, current_module_->env());
    }
    else {
      ret = eval(ret);
    }
    return ret;
  }
  catch (Exception& ex) {
    PSCM_ERROR("eval {0} error: {1}", code, ex.what());
    return Cell::ex(ex.what());
  }
}

Cell Scheme::eval_internal(SymbolTable *env, const UString code) {
  try {
    Parser parser(code);
    auto ret = parser.parse();
    if (ret.is_none()) {
      return Cell::none();
    }
    if (use_register_machine_) {
      ret = Evaluator(*this).eval(ret, env);
    }
    else {
      ret = eval(env, ret);
    }
    return ret;
  }
  catch (Exception& ex) {
    PSCM_ERROR("eval {0} error: {1}", code, ex.what());
    return Cell::ex(ex.what());
  }
}

void Scheme::eval_all(const UString& code, SourceLocation loc) {
  try {
    Parser parser(code);
    while (true) {

      auto expr = parser.next();
      if (expr.is_none()) {
        return;
      }
      if (use_register_machine_) {
        Evaluator(*this).eval(expr, current_module_->env());
      }
      else {
        eval(expr);
      }
    }
  }
  catch (Exception& ex) {
    PSCM_ERROR("eval {0} error: {1}", code, ex.what());
    PSCM_THROW_EXCEPTION(loc.to_string() + ", EVAL_ALL Error: " + code);
  }
}

bool Scheme::load(const UString& filename) {
  std::cout << "load: " << filename << std::endl;
  auto res = read_file(filename);
  if (!std::holds_alternative<UString>(res)){
    return false;
  }
  auto code = std::get<UString>(res);
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
    PSCM_ERROR("load file {0} error", filename);
    ex.print_stack_trace();
    return false;
  }
  return true;
}

Cell Scheme::eval_args(pscm::SymbolTable *env, pscm::Cell args,
                       SourceLocation loc PSCM_CXX20_MODULES_DEFAULT_ARG_COMPAT) {
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
    PSCM_TRACE("eval: {0}", expr.pretty_string());
    // PSCM_INFO("eval: {0}", expr);
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
      PSCM_DEBUG("proc {0} args: {1}", proc, args.pretty_string());
      PSCM_ASSERT(args.is_pair() || args.is_nil());
      auto f = proc.to_func();
      auto func_args = eval_args(env, args);
      auto ret = f->call(func_args);
      PSCM_DEBUG("proc {0} ret: {1}", expr.pretty_string(), ret.pretty_string());
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
        PSCM_DEBUG("op: {0}", op);
        PSCM_DEBUG("args: {0}", op_args);
        args = construct_apply_argl(op_args);
        // TODO:
        if (op.is_func()) {
          return op.to_func()->call(args);
        }
        else if (op.is_proc()) {
          expr = call_proc(env, op.to_proc(), args);
          // expr = cons(op, car(op_args));
          PSCM_DEBUG("new expr: {0}", expr);
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
        PSCM_TRACE("call macro: {0}", proc);
        expr = f->call(*this, env, args);
        PSCM_DEBUG("expand result pretty: {0}", expr.pretty_string());
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
      PSCM_ERROR("unsupported {0} from {1}", proc, proc.source_location());
      // repl();
      PSCM_THROW_EXCEPTION("unsupported");
    }
  }
}

Cell Scheme::eval(Cell expr) {
  PSCM_ASSERT(!envs_.empty());
  PSCM_ASSERT(current_module_);
  auto env = current_module_->env();
  auto ret = eval(env, expr);
  return ret;
}

Cell Scheme::lookup(SymbolTable *env, Cell expr, SourceLocation loc) {
  PSCM_ASSERT(env);
  PSCM_ASSERT(expr.is_sym());
  auto sym = expr.to_sym();
  PSCM_ASSERT(sym);
  bool has_sym = env->contains(sym);
  if (!has_sym) {
    sym->print_debug_info();
    PSCM_ERROR("env: {0} {1}", (void *)env, env->name());
    env->dump();
    // uncomment for debug
    // repl();
    PSCM_THROW_EXCEPTION("Unbound variable: " + sym->name());
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
  PSCM_DEBUG("body: {0}", body);
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

void Scheme::load_module(const UString& filename, Cell module_name) {
  auto old_module = current_module_;
  std::cout << "load: " << filename << std::endl;
  auto res = read_file(filename);
  if (!std::holds_alternative<UString>(res)){
    PSCM_THROW_EXCEPTION("load module error: " + Cell(module_name).to_string());
  }
  auto code = std::get<UString>(res);
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
    ex.print_stack_trace();
    PSCM_ERROR("load file {0} error", filename);
    PSCM_THROW_EXCEPTION("load module error: " + Cell(module_name).to_string());
  }
  current_module_ = old_module;
}

Module *Scheme::create_module(Cell name) {
  auto env = new SymbolTable(name.to_string(), root_derived_env_);
  return new Module(name, env);
}

void Scheme::repl() {
  in_repl_ = true;
  const auto path = "history.txt";

  // Enable the multi-line mode
  linenoise::SetMultiLine(true);

  // Set max length of the history
  linenoise::SetHistoryMaxLen(4);

  // Setup completion words every time when a user types
  linenoise::SetCompletionCallback([](const char *editBuffer, std::vector<std::string>& completions) {
    if (editBuffer[0] == 'd') {
      completions.push_back("define");
      completions.push_back("define-macro");
    }
  });

  // Load history
  linenoise::LoadHistory(path);

  while (true) {
    std::string line;
    auto quit = linenoise::Readline("pscm> ", line);
    if (quit) {
      break;
    }
    auto ret = eval(line.c_str());
    if (!ret.is_none()) {
      std::cout << ret.to_string() << std::endl;
    }
    // Add line to history
    linenoise::AddHistory(line.c_str());

    // Save history
    linenoise::SaveHistory(path);
  }
}

// useful for debug
PSCM_DEFINE_BUILTIN_MACRO(Scheme, "repl", Label::TODO) {
  while (true) {
    std::string line;
    auto quit = linenoise::Readline("pscm> ", line);
    if (quit) {
      break;
    }
    UString str(line.data());
    UIterator iter(str);
    auto expr = Parser(&iter).parse();
    auto ret = scm.eval(env, expr);
    if (!ret.is_none()) {
      std::cout << ret.to_string() << std::endl;
    }
  }
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_PROC(Scheme, "select") {
  PSCM_THROW_EXCEPTION("not implemented now");
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_PROC(Scheme, "scm-error") {
  PSCM_ERROR("scm-error: {0}", args);
  return Cell::none();
}

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "eval", Label::APPLY_EVAL) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (cdr(args).is_nil()) {
    auto expr = arg;
    auto ret = scm.eval(env, expr);
    return ret;
  }
  else {
    PSCM_THROW_EXCEPTION("wrong-number-of-args");
    auto env_arg = cadr(args);
    auto expr = arg;
    auto ret = scm.eval(env, expr);
    return ret;
  }
}

PSCM_DEFINE_BUILTIN_MACRO(Scheme, "primitive-eval", Label::APPLY_EVAL) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  if (cdr(args).is_nil()) {
    auto expr = arg;
    auto ret = scm.eval(env, expr);
    return ret;
  }
  else {
    PSCM_THROW_EXCEPTION("wrong-number-of-args");
    auto env_arg = cadr(args);
    auto expr = arg;
    auto ret = scm.eval(env, expr);
    return ret;
  }
}
} // namespace pscm
