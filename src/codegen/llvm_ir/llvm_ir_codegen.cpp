
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

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

llvm::ExitOnError exit_on_err;

int create_llvm_ir_add(llvm::LLVMContext& ctx, llvm::Module& m) {
  std::vector<llvm::Type *> func_args(2, llvm::Type::getInt64Ty(ctx));
  auto func_type = llvm::FunctionType::get(llvm::Type::getInt64Ty(ctx), func_args, false);
  auto func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, "pscm_jit_add2", &m);
  if (!func) {
    llvm::errs() << "create Function error"
                 << "\n";
    return -1;
  }
  std::vector<std::string> func_args_names;
  func_args_names.push_back("lhs");
  func_args_names.push_back("rhs");
  int idx = 0;
  for (auto& arg : func->args()) {
    arg.setName(func_args_names[idx++]);
  }

  auto basic_block = llvm::BasicBlock::Create(ctx, "entry", func);
  auto builder = std::make_unique<llvm::IRBuilder<>>(ctx);
  builder->SetInsertPoint(basic_block);
  auto ret = builder->CreateAdd(func->getArg(0), func->getArg(1), "addtmp");
  builder->CreateRet(ret);
  return 0;
}

std::optional<Cell> llvm_ir_codegen_and_run_jit(Cell expr) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  std::cout << "expr: " << expr.to_std_string() << std::endl;
  if (!expr.is_pair()) {
    return std::nullopt;
  }
  Cell fakeResult;
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
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("pscm jit", *ctx);

    auto jit_target_machine_builder = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jit_target_machine_builder) {
      llvm::errs() << "JITTargetMachineBuilder::detectHost() error"
                   << "\n";
      return std::nullopt;
    }

    auto default_data_layout = jit_target_machine_builder->getDefaultDataLayoutForTarget();
    if (!default_data_layout) {
      llvm::errs() << "getDefaultDataLayoutForTarget error"
                   << "\n";
      return std::nullopt;
    }

    module->setDataLayout(*default_data_layout);

    if (auto err = create_llvm_ir_add(*ctx, *module)) {
      llvm::errs() << "create mlir error"
                   << "\n";
      return std::nullopt;
    }

    auto self_executor_process_control = llvm::orc::SelfExecutorProcessControl::Create();
    if (!self_executor_process_control) {
      llvm::errs() << "Could not create SelfExecutorProcessControl"
                   << "\n";
      return std::nullopt;
    }

    auto execution_session = std::make_unique<llvm::orc::ExecutionSession>(std::move(*self_executor_process_control));
    auto get_section_memory_manager = []() {
      return std::make_unique<llvm::SectionMemoryManager>();
    };
    auto rt_dyld_object_linking_layer =
        std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(*execution_session, get_section_memory_manager);
    llvm::orc::IRCompileLayer ir_compile_layer(
        *execution_session, *rt_dyld_object_linking_layer,
        std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(*jit_target_machine_builder)));

    auto& main_jit_dylib = execution_session->createBareJITDylib("main");
    main_jit_dylib.addGenerator(exit_on_err(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(default_data_layout->getGlobalPrefix())));

    llvm::SMDiagnostic sm_diagnostic;
    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();
    llvm::errs() << *module << "\n";
    auto thread_safe_module = llvm::orc::ThreadSafeModule(std::move(module), std::move(llvm_ctx));

    exit_on_err(ir_compile_layer.add(main_jit_dylib, std::move(thread_safe_module)));

    auto func_sym = exit_on_err(execution_session->lookup({ &main_jit_dylib }, "_pscm_jit_add2"));
    auto func_ptr = func_sym.getAddress().toPtr<int64_t (*)(int64_t, int64_t)>();
    auto ret = func_ptr(n1->to_int(), n2->to_int());
    execution_session->endSession();
    return new pscm::Number(ret);
  }
  else {
    llvm::errs() << "not supported now"
                 << "\n";
    return std::nullopt;
  }
}

} // namespace pscm
