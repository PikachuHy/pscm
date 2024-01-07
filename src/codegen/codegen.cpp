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

#include "pscm/Number.h"
#include "pscm/Symbol.h"
#include "pscm/common_def.h"
#include "pscm/scm_utils.h"

#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/TargetParser/Host.h>

#include <iostream>
PSCM_INLINE_LOG_DECLARE("pscm.codegen");

namespace pscm {

mlir::Value create_i64(mlir::OpBuilder& builder, int64_t data) {
  auto dataType = builder.getI64Type();
  auto dataAttr = builder.getI64IntegerAttr(data);
  auto dataValue = builder.create<ConstantOp>(builder.getUnknownLoc(), dataType, dataAttr);
  return dataValue;
}

int link_to_executable(llvm::LLVMContext& ctx) {
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> opts = new clang::DiagnosticOptions;
  clang::DiagnosticsEngine diags(new clang::DiagnosticIDs, opts,
                                 new clang::TextDiagnosticPrinter(llvm::errs(), opts.get()));

  clang::driver::Driver d("clang", llvm::sys::getDefaultTargetTriple(), diags, "pscm compiler");
  std::vector<const char *> args = { "pscmc" };

  args.push_back("output.o");
  args.push_back("-o");
  args.push_back("a.out");
  args.push_back("-isysroot");
  args.push_back("/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk");
  args.push_back("-v");

  d.setCheckInputsExist(false);

  std::unique_ptr<clang::driver::Compilation> compilation;
  compilation.reset(d.BuildCompilation(args));

  if (!compilation) {
    return 1;
  }

  llvm::SmallVector<std::pair<int, const clang::driver::Command *>> failCommand;
  // compilation->ExecuteJobs(compilation->getJobs(), failCommand);

  d.ExecuteCompilation(*compilation, failCommand);
  if (failCommand.empty()) {
    llvm::outs() << "Done!\n";
  }
  else {
    llvm::errs() << "Linking failed!\n";
    return -1;
  }
  return 0;
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

  std::error_code ec;
  llvm::raw_fd_ostream dest("output.o", ec, llvm::sys::fs::OF_None);
  llvm::legacy::PassManager pass;
  auto fileType = llvm::CGFT_ObjectFile;
  if (machine.get()->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
    llvm::errs() << "TheTargetMachine can't emit a file of this type";
    return 1;
  }

  pass.run(*llvm_module);
  machine.get()->getTargetTriple();
  // FIXME
  // error: unable to execute command: Segmentation fault: 11
  // error: linker command failed due to signal (use -v to see invocation)
  // link_to_executable(ctx);
  // engine.get()->dumpToObjectFile("pscm.jit.o");
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

int create_mlir_add(mlir::MLIRContext& ctx, mlir::OwningOpRef<mlir::ModuleOp>& m, Cell a, Cell b) {
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
  PSCM_ASSERT(a.is_num());
  PSCM_ASSERT(b.is_num());
  auto num1 = a.to_num();
  auto num2 = b.to_num();
  auto lhs = create_i64(builder, num1->to_int());
  auto rhs = create_i64(builder, num2->to_int());

  auto ret = builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs);

  builder.create<PrintOp>(builder.getUnknownLoc(), ret);

  builder.create<ReturnOp>(builder.getUnknownLoc(), llvm::ArrayRef<mlir::Value>());

  if (failed(mlir::verify(*m))) {
    m->emitError("module verfification error");
    return -1;
  }

  return 0;
}

std::optional<Cell> mlir_codegen_and_run_jit(Cell expr) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<PSCMDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;

  std::cout << "expr: " << expr.to_std_string() << std::endl;
  if (!expr.is_pair()) {
    return std::nullopt;
  }
  // hardcode only suport (+ num1 num2)
  if (car(expr).is_sym() && *car(expr).to_sym() == "+"_sym) {
    auto num1 = cadr(expr);
    auto num2 = caddr(expr);
    if (!cdddr(expr).is_nil()) {
      return std::nullopt;
    }
    if (!num1.is_num() || !num2.is_num()) {
      return std::nullopt;
    }
    auto n1 = num1.to_num();
    auto n2 = num2.to_num();
    if (!n1->is_int() || !n2->is_int()) {
      return std::nullopt;
    }
    if (auto err = create_mlir_add(context, module, cadr(expr), caddr(expr))) {
      llvm::errs() << "create mlir error"
                   << "\n";
      return std::nullopt;
    }
  }
  else {
    llvm::errs() << "not supported now"
                 << "\n";
    return std::nullopt;
  }

  module->dump();
  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
    llvm::errs() << "applyPassManagerCLOptions error"
                 << "\n";
    return std::nullopt;
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
    return std::nullopt;
  }
  module->dump();
  if (auto err = run_jit(*module)) {
    llvm::errs() << "run mlir error"
                 << "\n";
    return std::nullopt;
  }
  return Cell::none();
}
} // namespace pscm
