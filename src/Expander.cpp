//
// Created by PikachuHy on 2023/3/26.
//
#ifdef PSCM_USE_CXX20_MODULES
#include "pscm/Logger.h"
#include "pscm/common_def.h"
import pscm;
import std;
import fmt;
#else
#include "pscm/Expander.h"
#include "pscm/Logger.h"
#include "pscm/Pair.h"
#include "pscm/Scheme.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <spdlog/fmt/fmt.h>
#include <string>
#endif
using namespace std::string_literals;
PSCM_INLINE_LOG_DECLARE("pscm.core.Expander");

namespace pscm {
Cell do_case(Cell item, Cell clause, Cell args) {
  if (!clause.is_pair()) {
    PSCM_THROW_EXCEPTION("Bad case clause " + clause.to_string() + " in expression " + args.to_string());
  }
  if (car(clause).is_sym() && car(clause) == "else"_sym) {
    return clause;
  }
  Cell new_clause = cons(list(new Symbol("member"), item, list(quote, car(clause))), cdr(clause));
  return new_clause;
}

Cell expand_case(Cell args) {
  auto key = car(args);
  auto clauses = cdr(args);
  Cell item = gensym();
  auto cond_clauses = cons(nil, nil);
  auto p = cond_clauses;
  while (!clauses.is_nil()) {
    auto clause = car(clauses);
    auto new_clause = do_case(item, clause, args);
    auto t = cons(new_clause, nil);
    p->second = t;
    p = t;
    clauses = cdr(clauses);
  }
  Cell cond_ = cons(new Symbol("cond"), cond_clauses->second);
  auto let_clause = list(list(list(item, key)), cond_);
  return expand_let(let_clause);
}

/*
(define-macro (let bindings . body)
  (define (varval v)
    (string->symbol (string-append (symbol->string v) "=")))

(define (named-let name bindings body)
     ((lambda (new-bindings)
       `(let ,(cons `(,name #f) new-bindings)
                   (set! ,name (lambda ,(map car bindings) . ,body))
                       (,name . ,(map car  new-bindings))))
          (map (lambda (b)
            `(,(varval (car b)) ,(cadr b))) bindings)))

        (if (symbol? bindings)
             (named-let bindings (car body) (cdr body))
      `((lambda ,(map car bindings) . ,body)  ,@(map cadr bindings))))
 */
Cell expand_named_let(Cell name, Cell bindings, Cell body) {
  auto var = map(car, bindings);
  auto arg = map(cadr, bindings);
  PSCM_DEBUG("var: {0}", var);
  PSCM_DEBUG("arg: {0}", arg);
  auto new_bindings = map(
      [](Cell expr, auto loc) {
        auto var = car(expr);
        auto init = cadr(expr);
        PSCM_ASSERT(var.is_sym());
        auto sym = var.to_sym();
        return list(new Symbol(sym->name() + "="), init);
      },
      bindings);
  PSCM_DEBUG("new-bindings: {0}", new_bindings);
  Cell let_init = cons(list(name, Cell::bool_false()), new_bindings);
  auto l2 = new Symbol("lambda");
  auto let_body = list(new Symbol("set!"), name, cons(l2, cons(map(car, bindings), body)));
  Cell let_body2 = cons(name, map(car, new_bindings));
  PSCM_DEBUG("let init: {0}", let_init);
  PSCM_DEBUG("let body: {0}", let_body);
  PSCM_DEBUG("let body2: {0}", let_body2);
  auto full_let = list(let_init, let_body, let_body2);
  PSCM_DEBUG("let: {0}", full_let);
  return expand_let(full_let);
}

Cell expand_let(Cell args) {
  // (let <bindings> <body>)
  // ((var1 init1) ...)
  // --->
  // ((lambda (var1 var2 ...) <body>) (init1 init2 ...))
  auto bindings = car(args);
  auto body = cdr(args);
  if (bindings.is_sym()) {
    return expand_named_let(bindings, car(body), cdr(body));
  }

  auto var = map(car, bindings);
  auto arg = map(cadr, bindings);

  auto a = cons(lambda, cons(var, body));
  Cell b = cons(a, arg);
  PSCM_DEBUG("let -> {0}", b.pretty_string());
  return b;
}

Cell expand_let_star(Cell args) {
  auto bindings = car(args);
  auto body = cdr(args);
  if (bindings.is_nil()) {
    return cons(cons(lambda, cons(nil, body)), nil);
  }
  auto arg = cons(car(bindings), nil);
  auto expanded_body = expand_let_star(cons(cdr(bindings), body));
  return expand_let(cons(arg, cons(expanded_body, nil)));
}

/*
(define-macro (letrec bindings . body)
  (let ((vars (map car  bindings))
        (vals (map cadr bindings)))

    `(let ,(map (lambda (var)
                  `(,var #f)) vars)

       ,@(map (lambda (var val)
                `(set! ,var ,val)) vars vals)
       . ,body)))
 */
Cell expand_letrec(Cell args) {
  auto bindings = car(args);
  auto body = cdr(args);

  auto vars = map(car, bindings);
  auto vals = map(cadr, bindings);

  auto let_var = map(
      [](auto item, auto loc) {
        return list(item, Cell::bool_false());
      },
      vars);

  auto update_let_var = map(
      [](auto item1, auto item2, auto loc) {
        return list(new Symbol("set!"), item1, item2);
      },
      vars, vals);

  PSCM_DEBUG("vars: {0}", vars.pretty_string());
  PSCM_DEBUG("vals: {0}", vals.pretty_string());
  PSCM_DEBUG("let_var: {0}", let_var.pretty_string());
  PSCM_DEBUG("update_let_var: {0}", update_let_var.pretty_string());
  auto p = update_let_var;
  if (update_let_var.is_nil()) {
    update_let_var = cons(nil, body);
  }
  else {
    while (p.is_pair()) {
      if (cdr(p).is_nil()) {
        auto p2 = p.to_pair();
        p2->second = body;
        break;
      }
      else {
        p = cdr(p);
      }
    }
  }

  //  PSCM_DEBUG("cons(update_let_var, body): {}", Cell(cons(update_let_var, body)));
  Cell expr = cons(let_var, update_let_var);
  PSCM_DEBUG("{0}", expr.pretty_string());
  return expand_let(expr);
}

Cell expand_do(Cell args) {
  auto bindings = car(args);
  auto test_and_expr = cadr(args);
  auto body = cddr(args);

  auto variables = map(car, bindings);
  auto inits = map(cadr, bindings);
  auto steps = map(
      [](auto expr, auto loc) {
        if (cddr(expr).is_nil()) {
          return car(expr);
        }
        else {
          return caddr(expr);
        }
      },
      bindings);

  auto test = car(test_and_expr);
  auto expr = cdr(test_and_expr);

  PSCM_DEBUG("var: {0}", variables);
  PSCM_DEBUG("init: {0}", inits);
  PSCM_DEBUG("step: {0}", steps);
  PSCM_DEBUG("test: {0}", test);
  PSCM_DEBUG("expr: {0}", expr);
  PSCM_DEBUG("body: {0}", body);
  Cell loop = gensym();
  {

    if (body.is_pair()) {
      auto new_body = cons(nil, nil);
      auto p = new_body;
      while (body.is_pair()) {
        auto new_pair = cons(car(body), nil);
        p->second = new_pair;
        p = new_pair;
        body = cdr(body);
      }
      p->second = list(cons(loop, steps));
      body = new_body->second;
    }
    else {
      body = list(cons(loop, steps));
    }
  }
  auto else_clause = cons(new Symbol("begin"), body);
  auto if_clause = Cell::none();
  if (!expr.is_nil()) {
    if_clause = cons(new Symbol("begin"), expr);
  }
  auto proc_body = list(&sym_if, test, if_clause, else_clause);
  PSCM_DEBUG("proc body: {0}", proc_body);
  auto proc_def = list(lambda, variables, proc_body);
  auto var_def = list(loop, proc_def);
  auto letrec_args = list(list(var_def), cons(loop, inits));
  PSCM_DEBUG("letrec: {0}", letrec_args);
  return expand_letrec(letrec_args);
}

bool QuasiQuotationExpander::is_constant(pscm::Cell expr) {
  if (expr.is_pair()) {
    auto op = car(expr);
    if (op.is_sym()) {
      auto sym = op.to_sym();
      auto val = env_->get_or(sym, Cell::none());
      return val == quote;
    }
    else {
      return op == quote;
    }
  }
  else {
    return !expr.is_sym();
  }
}

bool QuasiQuotationExpander::is_unquote(pscm::Cell expr) {
  return expr == "unquote"_sym;
}

bool QuasiQuotationExpander::is_quasiquote(pscm::Cell expr) {
  return expr == "quasiquote"_sym;
}

bool QuasiQuotationExpander::is_unquote_splicing(pscm::Cell expr) {
  return expr == "unquote-splicing"_sym;
}

bool QuasiQuotationExpander::is_list(pscm::Cell expr) {
  return expr == "list"_sym;
}

int QuasiQuotationExpander::length(Cell expr) {
  int len = 0;
  while (!expr.is_nil()) {
    len++;
    expr = cdr(expr);
  }
  return len;
}

Cell QuasiQuotationExpander::combine_skeletons(pscm::Cell left, pscm::Cell right, pscm::Cell expr) {
  if (is_constant(right) && is_constant(left)) {
    auto left_val = scm_.eval(env_, left);
    auto right_val = scm_.eval(env_, right);
    static Symbol sym_quote("quote");
    if (left_val.is_eqv(right_val).to_bool()) {
      return list(&sym_quote, expr);
    }
    else {
      return list(&sym_quote, cons(left_val, right_val));
    }
  }
  else if (right.is_nil()) {
    return list(new Symbol("list"), left);
  }
  else if (right.is_pair() && is_list(car(right))) {
    return cons(new Symbol("list"), cons(left, cdr(right)));
  }
  else {
    return list(new Symbol("cons"), left, right);
  }
}

Cell QuasiQuotationExpander::expand(pscm::Cell expr) {
  PSCM_DEBUG("`entry: {0}", expr.pretty_string());
  return expand(expr, 0);
}

Cell QuasiQuotationExpander::convert_vector_to_list(const Cell::Vec& vec) {
  auto ret = cons(nil, nil);
  auto p = ret;
  for (const auto& item : vec) {
    auto new_pair = cons(item, nil);
    p->second = new_pair;
    p = new_pair;
  }
  return ret->second;
}

Cell QuasiQuotationExpander::expand(pscm::Cell expr, int nesting) {
  PSCM_DEBUG("expand: {0}, nesting: {1}", expr.pretty_string(), nesting);
  if (expr.is_vec()) {
    PSCM_DEBUG("expr: {0}", expr);
    auto l = convert_vector_to_list(*expr.to_vec());
    PSCM_DEBUG("l: {0}", l);
    auto new_expr = expand(l, nesting);
    PSCM_DEBUG("new_expr: {0}", new_expr);
    auto ret = list(new Symbol("apply"), new Symbol("vector"), new_expr);
    PSCM_DEBUG("expand ret: {0}", ret);
    return ret;
  }
  else if (!expr.is_pair()) {
    if (is_constant(expr)) {
      return expr;
    }
    else {
      return list(quote, expr);
    }
  }
  else if (is_unquote(car(expr)) && length(expr) == 2) {
    if (nesting == 0) {
      return cadr(expr);
    }
    else {
      auto new_right = expand(cdr(expr), nesting - 1);
      PSCM_DEBUG("new right: {0}", new_right);
      static Symbol sym_unquote("unquote");
      return combine_skeletons(list(quote, &sym_unquote), new_right, expr);
    }
  }
  else if (is_quasiquote(car(expr)) && length(expr) == 2) {
    auto new_right = expand(cdr(expr), nesting + 1);
    static Symbol sym_quasiquote("quasiquote");
    return combine_skeletons(list(quote, &sym_quasiquote), new_right, expr);
  }
  else if (car(expr).is_pair() && is_unquote_splicing(caar(expr)) && length(car(expr)) == 2) {
    if (nesting == 0) {
      auto new_expr = expand(cdr(expr), nesting);
      auto ret = list(new Symbol("append"), cadr(car(expr)), new_expr);
      PSCM_DEBUG(",@: {0}", ret);
      return ret;
    }
    else {
      auto new_left = expand(car(expr), nesting - 1);
      auto new_right = expand(cdr(expr), nesting);
      return combine_skeletons(new_left, new_right, expr);
    }
  }
  else {
    auto new_left = expand(car(expr), nesting);
    auto new_right = expand(cdr(expr), nesting);
    return combine_skeletons(new_left, new_right, expr);
  }
}
} // namespace pscm