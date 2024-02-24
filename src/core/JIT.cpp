#include "JIT.h"
#include "Runtime.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include <pscm/common_def.h>

namespace pscm::core {
PSCM_INLINE_LOG_DECLARE("pscm.core.JIT");

JIT::JIT(std::unique_ptr<llvm::orc::ExecutionSession> es, llvm::orc::JITTargetMachineBuilder jtmb, llvm::DataLayout dl)
    : es_(std::move(es))
    , dl_(std::move(dl))
    , mangle_(*this->es_, this->dl_)
    , object_layer_(*this->es_,
                    []() {
                      return std::make_unique<llvm::SectionMemoryManager>();
                    })
    , compile_layer_(*this->es_, object_layer_, std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(jtmb)))
    , main_jd_(this->es_->createBareJITDylib("<main>")) {
  main_jd_.addGenerator(
      llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(dl_.getGlobalPrefix())));

  void *malloc_addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("malloc");
  if (!malloc_addr) {
    PSCM_THROW_EXCEPTION("malloc not found");
  }
  auto malloc_sym = llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr(llvm::pointerToJITTargetAddress(malloc_addr)),
                                                 llvm::JITSymbolFlags::Exported);
  llvm::cantFail(main_jd_.define(llvm::orc::absoluteSymbols({
      { es_->intern("malloc"), malloc_sym },
      { es_->intern("car_array[integer]"),
       llvm::orc::ExecutorSymbolDef(llvm::orc::ExecutorAddr(llvm::pointerToJITTargetAddress(car_array)),
       llvm::JITSymbolFlags::Exported) }
  })));
}
} // namespace pscm::core