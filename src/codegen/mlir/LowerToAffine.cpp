#include "pscm/codegen/mlir/Passes.h"
using namespace mlir;

#include "mlir/IR/BuiltinDialect.h"
#include "pscm/codegen/mlir/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

namespace {
struct FuncOpLowering : public OpConversionPattern<pscm::FuncOp> {
  using OpConversionPattern<pscm::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pscm::FuncOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const final {
    if (op.getName() != "main")
      return failure();
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpLowering : public OpRewritePattern<pscm::ReturnOp> {
  using OpRewritePattern<pscm::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pscm::ReturnOp op, PatternRewriter& rewriter) const final {
    if (op.hasOperand())
      return failure();
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

} // namespace

struct PSCMToAffineLoweringPass : public PassWrapper<PSCMToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PSCMToAffineLoweringPass)

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override final {

    ConversionTarget target(getContext());

    target.addLegalDialect<affine::AffineDialect, BuiltinDialect, arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<FuncOpLowering, ReturnOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

namespace pscm {

std::unique_ptr<mlir::Pass> createLowerToAffinePass() {
  return std::make_unique<PSCMToAffineLoweringPass>();
}
} // namespace pscm
