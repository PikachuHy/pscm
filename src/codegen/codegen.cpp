#include "pscm/codegen/mlir/Passes.h"

using namespace mlir;

#include "pscm/codegen/codegen.h"
#include "pscm/codegen/mlir/Dialect.h"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/ScopedHashTable.h"

#include <iostream>

namespace pscm {

mlir::Value create_i64(mlir::OpBuilder& builder, int64_t data) {
  auto dataType = builder.getI64Type();
  auto dataAttr = builder.getI64IntegerAttr(data);
  auto dataValue = builder.create<ConstantOp>(builder.getUnknownLoc(), dataType, dataAttr);
  return dataValue;
}

int run_jit(mlir::ModuleOp m) {
  mlir::registerBuiltinDialectTranslation(*m->getContext());
  mlir::registerLLVMDialectTranslation(*m->getContext());
  llvm::LLVMContext ctx;
  auto llvm_module = mlir::translateModuleToLLVMIR(m, ctx);
  if (!llvm_module) {
    llvm::errs() << "Failed to emit LLVM IR "
                 << "\n";
    return -1;
  }
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!builder) {
    llvm::errs() << "Could not create JITTargetMachineBuilder"
                 << "\n";
    return -1;
  }
  auto machine = builder->createTargetMachine();
  if (!machine) {
    llvm::errs() << "Could not create TargetMachine"
                 << "\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvm_module.get(), machine.get().get());
  auto opt = mlir::makeOptimizingTransformer(0, 0, nullptr);
  if (auto err = opt(llvm_module.get())) {
    llvm::errs() << "Failed to optimize LLVM IR"
                 << "\n";
    return -1;
  }
  // llvm::errs() << *llvm_module << "\n";

  mlir::ExecutionEngineOptions engine_options;
  engine_options.transformer = opt;

  auto engine = mlir::ExecutionEngine::create(m, engine_options);

  if (!engine) {
    llvm::errs() << "Failed to construct an execution engine"
                 << "\n";
    return -1;
  }
  auto invocationResult = engine.get()->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed"
                 << "\n";
    return -1;
  }
  return 0;
}

int create_mlir(mlir::MLIRContext& ctx, mlir::OwningOpRef<mlir::ModuleOp>& m) {
  mlir::OpBuilder builder(&ctx);
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  m = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(m->getBody());
  llvm::SmallVector<mlir::Type, 4> argTypes;
  argTypes.reserve(0);
  auto mainFuncType = builder.getFunctionType(argTypes, std::nullopt);
  auto mainFunc = builder.create<FuncOp>(builder.getUnknownLoc(), "main", mainFuncType);
  mlir::Block& entryBlock = mainFunc.front();
  builder.setInsertionPointToStart(&entryBlock);

  auto lhs = create_i64(builder, 1);
  auto rhs = create_i64(builder, 2);

  auto ret = builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs);

  builder.create<PrintOp>(builder.getUnknownLoc(), ret);

  builder.create<ReturnOp>(builder.getUnknownLoc(), llvm::ArrayRef<mlir::Value>());

  if (failed(mlir::verify(*m))) {
    m->emitError("module verfification error");
    return -1;
  }

  return 0;
}

Cell mlir_codegen_and_run_jit(Cell expr) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<PSCMDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;

  if (auto err = create_mlir(context, module)) {
    llvm::errs() << "create mlir error"
                 << "\n";
    return Cell::bool_false();
  }
  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
    llvm::errs() << "applyPassManagerCLOptions error"
                 << "\n";
    return Cell::bool_false();
  }
  mlir::OpPassManager& optPM = pm.nest<pscm::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
  pm.addPass(createLowerToAffinePass());
  pm.addPass(createLowerToLLVMPass());
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "run pass error"
                 << "\n";
    return Cell::bool_false();
  }
  // module->dump();
  if (auto err = run_jit(*module)) {
    llvm::errs() << "run mlir error"
                 << "\n";
    return Cell::bool_false();
  }
  return Cell::none();
}
} // namespace pscm
