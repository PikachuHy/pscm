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
} // namespace pscm