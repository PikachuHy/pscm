#pragma once
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorSymbolDef.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include <memory>

namespace pscm::core {

class JIT {
public:
  JIT(std::unique_ptr<llvm::orc::ExecutionSession> es, llvm::orc::JITTargetMachineBuilder jtmb, llvm::DataLayout dl);

  ~JIT() {
    if (auto err = es_->endSession(); err) {
      es_->reportError(std::move(err));
    }
  }

  [[nodiscard]] static llvm::Expected<std::unique_ptr<JIT>> create() {
    auto epc = llvm::orc::SelfExecutorProcessControl::Create();
    if (!epc) {
      return epc.takeError();
    }
    auto es = std::make_unique<llvm::orc::ExecutionSession>(std::move(*epc));
    llvm::orc::JITTargetMachineBuilder jtmb(es->getExecutorProcessControl().getTargetTriple());
    auto dl = jtmb.getDefaultDataLayoutForTarget();
    if (!dl) {
      return dl.takeError();
    }
    return std::make_unique<JIT>(std::move(es), std::move(jtmb), std::move(*dl));
  }

  [[nodiscard]] const llvm::DataLayout& data_layout() const {
    return dl_;
  }

  [[nodiscard]] llvm::orc::JITDylib& main_jit_dylib() {
    return main_jd_;
  }

  [[nodiscard]] llvm::Error add_module(llvm::orc::ThreadSafeModule tsm, llvm::orc::ResourceTrackerSP rt = nullptr) {
    if (!rt) {
      rt = main_jd_.getDefaultResourceTracker();
    }
    return compile_layer_.add(rt, std::move(tsm));
  }

  [[nodiscard]] llvm::Expected<llvm::orc::ExecutorSymbolDef> lookup(llvm::StringRef name) {
    return es_->lookup({ &main_jd_ }, mangle_(name.str()));
  }

private:
  std::unique_ptr<llvm::orc::ExecutionSession> es_;
  llvm::DataLayout dl_;
  llvm::orc::MangleAndInterner mangle_;
  llvm::orc::RTDyldObjectLinkingLayer object_layer_;
  llvm::orc::IRCompileLayer compile_layer_;
  llvm::orc::JITDylib& main_jd_;
};
} // namespace pscm::core
