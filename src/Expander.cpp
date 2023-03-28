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
} // namespace pscm