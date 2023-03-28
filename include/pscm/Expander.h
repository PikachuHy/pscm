//
// Created by PikachuHy on 2023/3/26.
//

#pragma once
#include "pscm/Cell.h"

namespace pscm {
Cell expand_let(Cell args);
Cell expand_let_star(Cell args);
Cell expand_letrec(Cell args);
Cell expand_case(Cell args);

class Expander {};

} // namespace pscm
