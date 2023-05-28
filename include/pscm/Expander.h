//
// Created by PikachuHy on 2023/3/26.
//

#pragma once
#include "pscm/Cell.h"
#include "pscm/SchemeProxy.h"

namespace pscm {
Cell expand_let(Cell args);
Cell expand_let_star(Cell args);
Cell expand_letrec(Cell args);
Cell expand_case(Cell args);
Cell expand_do(Cell args);
class Scheme;

class Expander {};

class QuasiQuotationExpander {
public:
  QuasiQuotationExpander(SchemeProxy scm, SymbolTable *env)
      : scm_(scm)
      , env_(env) {
  }

  Cell expand(Cell expr);

private:
  [[nodiscard]] bool is_constant(Cell expr);
  [[nodiscard]] bool is_unquote(Cell expr);
  [[nodiscard]] bool is_quasiquote(Cell expr);
  [[nodiscard]] bool is_unquote_splicing(Cell expr);
  [[nodiscard]] bool is_list(Cell expr);
  [[nodiscard]] int length(Cell expr);
  [[nodiscard]] Cell convert_vector_to_list(const Cell::Vec& vec);
  [[nodiscard]] Cell combine_skeletons(Cell left, Cell right, Cell expr);
  [[nodiscard]] Cell expand(Cell expr, int nesting);

private:
  SchemeProxy scm_;
  SymbolTable *env_;
};
} // namespace pscm
