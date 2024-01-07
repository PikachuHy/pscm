#pragma once
#include "pscm/Cell.h"
#include <optional>

namespace pscm {
std::optional<Cell> mlir_codegen_and_run_jit(Cell expr);
}