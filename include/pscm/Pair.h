//
// Created by PikachuHy on 2023/2/25.
//

#pragma once
#include "pscm/Cell.h"

namespace pscm {
class Pair {
public:
  Cell first;
  Cell second;
  bool operator==(const Pair& rhs) const;
  bool operator!=(const Pair& rhs) const;
};

inline Pair *cons(Cell a, Cell b) {
  return new Pair{ a, b };
}

Cell car(Cell c, SourceLocation loc = {});
Cell cdr(Cell c, SourceLocation loc = {});
Cell cdar(Cell c, SourceLocation loc = {});
Cell cadr(Cell c, SourceLocation loc = {});
Cell cddr(Cell c, SourceLocation loc = {});
} // namespace pscm
