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
Cell caar(Cell c, SourceLocation loc = {});
Cell caaar(Cell c, SourceLocation loc = {});
Cell cdr(Cell c, SourceLocation loc = {});
Cell cdar(Cell c, SourceLocation loc = {});
Cell cadr(Cell c, SourceLocation loc = {});
Cell cadar(Cell c, SourceLocation loc = {});
Cell caadr(Cell c, SourceLocation loc = {});
Cell cdadr(Cell c, SourceLocation loc = {});
Cell cddr(Cell c, SourceLocation loc = {});
Cell cdddr(Cell c, SourceLocation loc = {});
Cell caddr(Cell c, SourceLocation loc = {});
Cell caddar(Cell c, SourceLocation loc = {});
Cell cadddr(Cell c, SourceLocation loc = {});

[[nodiscard]] int list_length(Cell expr);
void set_car(Cell list, Cell val);
void set_cdr(Cell list, Cell val);
} // namespace pscm
