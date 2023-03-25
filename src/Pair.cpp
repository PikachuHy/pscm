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

bool Pair::operator==(const Pair& rhs) const {
  return first == rhs.first && second == rhs.second;
}

bool Pair::operator!=(const Pair& rhs) const {
  return !(rhs == *this);
}
} // namespace pscm