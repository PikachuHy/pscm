//
// Created by PikachuHy on 2023/3/26.
//

#include "pscm/Expander.h"
#include "pscm/Pair.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"

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
Cell expand_let(Cell args) {
  // (let <bindings> <body>)
  // ((var1 init1) ...)
  // --->
  // ((lambda (var1 var2 ...) <body>) (init1 init2 ...))
  auto bindings = car(args);
  auto body = cdr(args);

  auto var = map(car, bindings);
  auto arg = map(cadr, bindings);

  auto a = cons(lambda, cons(var, body));
  auto b = cons(a, arg);
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

  SPDLOG_INFO("vars: {}", vars);
  SPDLOG_INFO("vals: {}", vals);
  SPDLOG_INFO("let_var: {}", let_var);
  SPDLOG_INFO("update_let_var: {}", update_let_var);
  auto p = update_let_var;
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
  //  SPDLOG_INFO("cons(update_let_var, body): {}", Cell(cons(update_let_var, body)));
  Cell expr = cons(let_var, update_let_var);
  SPDLOG_INFO("{}", expr);
  return expand_let(expr);
}

} // namespace pscm