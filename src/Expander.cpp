//
// Created by PikachuHy on 2023/3/26.
//

#include "pscm/Expander.h"
#include "pscm/Pair.h"
#include "pscm/Scheme.h"
#include "pscm/Symbol.h"
#include "pscm/SymbolTable.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"
#include <string>
using namespace std::string_literals;

namespace pscm {
Cell do_case(auto item, Cell clause, auto args) {
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
  SPDLOG_DEBUG("var: {}", var);
  SPDLOG_DEBUG("arg: {}", arg);
  auto new_bindings = map(
      [](Cell expr, auto loc) {
        auto var = car(expr);
        auto init = cadr(expr);
        PSCM_ASSERT(var.is_sym());
        auto sym = var.to_sym();
        return list(new Symbol(std::string(sym->name()) + "="s), init);
      },
      bindings);
  SPDLOG_DEBUG("new-bindings: {}", new_bindings);
  Cell let_init = cons(list(name, Cell::bool_false()), new_bindings);
  auto l2 = new Symbol("lambda");
  auto let_body = list(new Symbol("set!"), name, cons(l2, cons(map(car, bindings), body)));
  Cell let_body2 = cons(name, map(car, new_bindings));
  SPDLOG_DEBUG("let init: {}", let_init);
  SPDLOG_DEBUG("let body: {}", let_body);
  SPDLOG_DEBUG("let body2: {}", let_body2);
  auto full_let = list(let_init, let_body, let_body2);
  SPDLOG_DEBUG("let: {}", full_let);
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
  SPDLOG_DEBUG("let -> {}", b.pretty_string());
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

  SPDLOG_DEBUG("vars: {}", vars.pretty_string());
  SPDLOG_DEBUG("vals: {}", vals.pretty_string());
  SPDLOG_DEBUG("let_var: {}", let_var.pretty_string());
  SPDLOG_DEBUG("update_let_var: {}", update_let_var.pretty_string());
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

  //  SPDLOG_DEBUG("cons(update_let_var, body): {}", Cell(cons(update_let_var, body)));
  Cell expr = cons(let_var, update_let_var);
  SPDLOG_DEBUG("{}", expr.pretty_string());
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

  SPDLOG_DEBUG("var: {}", variables);
  SPDLOG_DEBUG("init: {}", inits);
  SPDLOG_DEBUG("step: {}", steps);
  SPDLOG_DEBUG("test: {}", test);
  SPDLOG_DEBUG("expr: {}", expr);
  SPDLOG_DEBUG("body: {}", body);
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
  SPDLOG_DEBUG("proc body: {}", proc_body);
  auto proc_def = list(lambda, variables, proc_body);
  auto var_def = list(loop, proc_def);
  auto letrec_args = list(list(var_def), cons(loop, inits));
  SPDLOG_DEBUG("letrec: {}", letrec_args);
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
  SPDLOG_DEBUG("`entry: {}", expr.pretty_string());
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
  SPDLOG_DEBUG("expand: {}, nesting: {}", expr.pretty_string(), nesting);
  if (expr.is_vec()) {
    SPDLOG_DEBUG("expr: {}", expr);
    auto l = convert_vector_to_list(*expr.to_vec());
    SPDLOG_DEBUG("l: {}", l);
    auto new_expr = expand(l, nesting);
    SPDLOG_DEBUG("new_expr: {}", new_expr);
    auto ret = list(new Symbol("apply"), new Symbol("vector"), new_expr);
    SPDLOG_DEBUG("expand ret: {}", ret);
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
      SPDLOG_DEBUG("new right: {}", new_right);
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
      SPDLOG_DEBUG(",@: {}", ret);
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