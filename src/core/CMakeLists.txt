set(CMAKE_MODULE_PATH /usr/local/opt/llvm/lib/cmake/clang /usr/local/opt/llvm/lib/cmake/lld /usr/local/opt/llvm/lib/cmake/llvm /usr/local/opt/llvm/lib/cmake/mlir)
set(CMAKE_PREFIX_PATH /usr/local/opt/llvm/lib/cmake/clang /usr/local/opt/llvm/lib/cmake/lld /usr/local/opt/llvm/lib/cmake/llvm /usr/local/opt/llvm/lib/cmake/mlir)
find_package(LLVM REQUIRED)
find_package(MLIR REQUIRED)
find_package(Clang REQUIRED)
add_library(pscm_core)
target_include_directories(pscm_core PUBLIC /usr/local/opt/llvm/include)
target_sources(pscm_core PUBLIC
    Value.cpp
    Value.h
    Parser.cpp
    Parser.h
    Evaluator.cpp
    Evaluator.h
    SymbolTable.cpp
    SymbolTable.h
    Procedure.cpp
    Procedure.h
    JIT.cpp
    JIT.h
    Scheme.cpp
    Scheme.h
    Mangler.cpp
    Mangler.h
    Runtime.cpp
    Runtime.h
)

target_link_libraries(pscm_core PUBLIC 
    clangDriver
    clangTooling
    clangFrontend
    pscm
)

target_compile_definitions(pscm_core PUBLIC PSCM_ENABLE_LLVM_CODEGEN)
