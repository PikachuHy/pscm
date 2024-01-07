#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pscm {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
} // namespace pscm
