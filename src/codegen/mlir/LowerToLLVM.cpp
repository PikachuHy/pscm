#include "pscm/codegen/mlir/Passes.h"

using namespace mlir;
#include "pscm/codegen/mlir/Dialect.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace pscm;

namespace {
struct ConstantOpLowering : public OpRewritePattern<ConstantOp> {
  using OpRewritePattern<ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConstantOp op, PatternRewriter& rewriter) const final {
    auto value = op.getValueAttr();
    auto loc = op.getLoc();
    auto new_value = rewriter.create<LLVM::ConstantOp>(loc, value);
    rewriter.replaceOp(op, new_value);
    return success();
  }
};

struct AddOpLowering : public ConversionPattern {
  explicit AddOpLowering(MLIRContext *ctx)
      : ConversionPattern(AddOp::getOperationName(), 1, ctx) {
  }

  LogicalResult matchAndRewrite(Operation *op, llvm::ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    typename AddOp::Adaptor binaryAdaptor(operands);
    auto loc = op->getLoc();
    auto new_op = rewriter.create<LLVM::AddOp>(loc, binaryAdaptor.getLhs(), binaryAdaptor.getRhs());
    rewriter.replaceOp(op, new_op);
    return success();
  }
};

class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(pscm::PrintOp::getOperationName(), 1, context) {
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst =
        getOrCreateGlobalString(loc, rewriter, "frmt_spec", StringRef("jit: %d \0", 8), parentModule);
    Value newLineCst = getOrCreateGlobalString(loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

    rewriter.create<func::CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                  llvm::ArrayRef<Value>({ formatSpecifierCst, operands[0] }));

    rewriter.create<func::CallOp>(loc, printfRef, rewriter.getIntegerType(32), llvm::ArrayRef<Value>({ newLineCst }));

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter& rewriter, ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder& builder, StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true, LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(), builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
                                       globalPtr, ArrayRef<Value>({ cst0, cst0 }));
  }
};
} // namespace

namespace pscm {
struct PSCMToLLVMLoweringPass : public PassWrapper<PSCMToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PSCMToLLVMLoweringPass)

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() override final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    patterns.add<ConstantOpLowering, AddOpLowering, PrintOpLowering>(&getContext());

    auto module = getOperation();

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      llvm::errs() << "apply full conversion error"
                   << "\n";
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<PSCMToLLVMLoweringPass>();
}
} // namespace pscm