//
// Created by PikachuHy on 2023/2/25.
//

#include "pscm/Pair.h"
#include "pscm/Cell.h"
#include "pscm/Exception.h"
#include "pscm/common_def.h"

namespace pscm {
Cell car(Cell c, SourceLocation loc) {
  if (!c.is_pair()) {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Cell is not Pair: " + c.to_string());
  }
  auto p = c.to_pair();
  return p->first;
}

Cell caar(Cell c, SourceLocation loc) {
  return car(car(c, loc), loc);
}

Cell caaar(Cell c, SourceLocation loc) {
  return car(car(car(c, loc), loc), loc);
}

Cell cdr(Cell c, SourceLocation loc) {
  if (!c.is_pair()) {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Cell is not Pair: " + c.to_string());
  }
  auto p = c.to_pair();
  return p->second;
}

Cell cdar(Cell c, SourceLocation loc) {
  return cdr(car(c, loc), loc);
}

Cell cadr(Cell c, SourceLocation loc) {
  return car(cdr(c, loc), loc);
}

Cell caadr(Cell c, SourceLocation loc) {
  return car(car(cdr(c, loc), loc), loc);
}

Cell cdadr(Cell c, SourceLocation loc) {
  return cdr(car(cdr(c, loc), loc), loc);
}

Cell cddr(Cell c, SourceLocation loc) {
  if (!c.is_pair()) {
    PSCM_THROW_EXCEPTION(loc.to_string() + ", Cell is not Pair: " + c.to_string());
  }
  auto p = c.to_pair();
  if (!p->second.is_pair()) {
    PSCM_THROW_EXCEPTION("Cell is not Pair: " + p->second.to_string() + " " + loc.to_string());
  }
  p = p->second.to_pair();
  return p->second;
}

Cell cdddr(Cell c, SourceLocation loc) {
  return cdr(cddr(c, loc), loc);
}

Cell caddr(Cell c, SourceLocation loc) {
  return car(cdr(cdr(c, loc), loc), loc);
}

Cell cadddr(Cell c, SourceLocation loc) {
  return car(cdr(cdr(cdr(c, loc), loc), loc), loc);
}

bool Pair::operator==(const Pair& rhs) const {
  return first == rhs.first && second == rhs.second;
}

bool Pair::operator!=(const Pair& rhs) const {
  return !(rhs == *this);
}

int list_length(Cell expr) {
  int len = 0;
  while (!expr.is_nil()) {
    len++;
    expr = cdr(expr);
  }
  return len;
}

Cell proc_cons(Cell args) {
  auto a = car(args);
  auto b = cadr(args);
  return cons(a, b);
}

Cell proc_car(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return car(arg);
}

Cell proc_caar(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return caar(arg);
}

Cell proc_cdr(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cdr(arg);
}

Cell proc_cdar(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cdar(arg);
}

Cell proc_cadr(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cadr(arg);
}

Cell proc_cddr(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return cddr(arg);
}

Cell proc_caddr(Cell args) {
  PSCM_ASSERT(args.is_pair());
  auto arg = car(args);
  return caddr(arg);
}

} // namespace pscm