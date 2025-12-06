cmake_minimum_required(VERSION 3.26)
project(pscm VERSION 0.3.0 LANGUAGES C CXX)
option(PSCM_ENABLE_MLIR_CODEGEN "enable codegen with MLIR" OFF)
set(PSCM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PSCM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()
add_library(pscm STATIC)
if (EMSCRIPTEN)
    target_link_options(pscm PUBLIC "-sUSE_ICU=1")
    target_compile_options(pscm PUBLIC "-sUSE_ICU=1")
elseif (WIN32)
    find_package(icu REQUIRED i18n uc io)
    target_link_libraries(pscm PUBLIC icu)
else ()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ICU REQUIRED icu-i18n icu-uc icu-io IMPORTED_TARGET)
    target_link_libraries(pscm PUBLIC PkgConfig::ICU)
endif ()

option(USE_CPP_STD_COMPAT "use c++14 std library compat" ON)
find_package(Git REQUIRED)
message(STATUS "Git: ${GIT_EXECUTABLE}")
set(GIT_HASH "unknown")
set(GIT_BRANCH "unknown")
execute_process(
        COMMAND ${GIT_EXECUTABLE} log -1 --pretty=format:%H
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}

)
execute_process(
        COMMAND ${GIT_EXECUTABLE} symbolic-ref --short -q HEAD
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
message(STATUS "GIT_BRANCH: ${GIT_BRANCH}")
message(STATUS "GIT_HASH: ${GIT_HASH}")
#configure_file(
#        ${CMAKE_CURRENT_SOURCE_DIR}/include/pscm/version.h.in
#        ${CMAKE_CURRENT_BINARY_DIR}/generate/pscm/version.h
#        @ONLY
#)
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/generate/src/version.cpp
        @ONLY
)
# use ccache
option(USE_CCACHE "use ccache" OFF)
if (USE_CCACHE)
    find_program(CCACHE_FOUND ccache)
    if (CCACHE_FOUND)
        set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
        set(CMAKE_C_COMPILER_LAUNCHER ccache)
        message(STATUS "ccache found")
    else ()
        message(WARNING "ccache not found")
    endif ()
endif ()
add_subdirectory(3rd)
target_include_directories(pscm PUBLIC include)
target_include_directories(pscm PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generate)
target_include_directories(pscm PUBLIC 3rd/UniversalStacktrace)
target_include_directories(pscm PUBLIC 3rd/cpp-linenoise)
target_include_directories(pscm PUBLIC 3rd/mscharconv/include)
if (USE_CPP_STD_COMPAT)
    target_include_directories(pscm PUBLIC 3rd/variant/include)
    target_include_directories(pscm PUBLIC 3rd/optional/include)
    target_include_directories(pscm PUBLIC 3rd/filesystem/include)
    target_include_directories(pscm PUBLIC 3rd/string-view-lite/include)
endif ()

file(GLOB PSCM_SRCS "src/*.cpp")
target_sources(pscm PRIVATE ${PSCM_SRCS})
if (PSCM_ENABLE_MLIR_CODEGEN)
    message(STATUS "Enable codegen with MLIR")
    target_compile_definitions(pscm PRIVATE PSCM_ENABLE_MLIR_CODEGEN)
    file(GLOB CODEGEN_SRCS src/codegen/*.cpp src/codegen/mlir/*.cpp src/codegen/llvm_ir/*.cpp)
    target_sources(pscm PRIVATE ${CODEGEN_SRCS})
    set(CMAKE_MODULE_PATH /usr/local/opt/llvm/lib/cmake/clang /usr/local/opt/llvm/lib/cmake/lld /usr/local/opt/llvm/lib/cmake/llvm /usr/local/opt/llvm/lib/cmake/mlir)
    set(CMAKE_PREFIX_PATH /usr/local/opt/llvm/lib/cmake/clang /usr/local/opt/llvm/lib/cmake/lld /usr/local/opt/llvm/lib/cmake/llvm /usr/local/opt/llvm/lib/cmake/mlir)
    find_package(LLVM REQUIRED)
    find_package(MLIR REQUIRED)
    find_package(Clang REQUIRED)
    include(TableGen)
    include(AddMLIR)
    include_directories(/usr/local/opt/llvm/include)
    set(LLVM_TARGET_DEFINITIONS include/pscm/codegen/mlir/Ops.td)
    mlir_tablegen(generate/Ops.h.inc -gen-op-decls)
    mlir_tablegen(generate/Ops.cpp.inc -gen-op-defs)
    mlir_tablegen(generate/Dialect.h.inc -gen-dialect-decls)
    mlir_tablegen(generate/Dialect.cpp.inc -gen-dialect-defs)
    add_public_tablegen_target(pscm-inc-gen)
    add_dependencies(pscm pscm-inc-gen)
    get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
    get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
    target_link_libraries(pscm PRIVATE
            ${dialect_libs}
            ${conversion_libs}
            ${extension_libs}

            MLIRAnalysis
            MLIRBuiltinToLLVMIRTranslation
            MLIRCastInterfaces
            MLIRCallInterfaces
            MLIRExecutionEngine
            MLIRLLVMToLLVMIRTranslation
            MLIRMemRefDialect
            MLIRIR
            MLIRParser
            MLIRPass
            MLIRSideEffectInterfaces
            MLIRSupport
            MLIRTargetLLVMIRExport
            MLIRTransforms

            clangDriver
            clangTooling
            clangFrontend
    )
endif ()
target_sources(pscm PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generate/src/version.cpp
        src/logger/Logger.cpp
        src/logger/Appender.cpp
        src/misc/SourceLocation.cpp
        src/icu/ICUCompat.cpp
)
target_link_libraries(pscm PUBLIC spdlog::spdlog)
if (EMSCRIPTEN)
    target_compile_definitions(pscm PUBLIC WASM_PLATFORM)
    target_link_options(pscm PUBLIC "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap")
    target_link_options(pscm PUBLIC "-sSTACK_SIZE=6553600")
    target_link_options(pscm PUBLIC "-sNO_DISABLE_EXCEPTION_CATCHING")
    add_subdirectory(wasm)
else ()
    add_executable(pscm_main main.cpp)
    if (PSCM_ENABLE_LLVM_CODEGEN)
        add_subdirectory(src/core)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "15.7" CACHE STRING "Minimum OS X deployment version" FORCE)
        target_link_libraries(pscm_main PRIVATE pscm_core)
        target_include_directories(pscm_main PRIVATE src)
    else ()
        set(CMAKE_OSX_DEPLOYMENT_TARGET "15.7" CACHE STRING "Minimum OS X deployment version" FORCE)
        target_link_libraries(pscm_main PRIVATE pscm)
    endif ()
endif ()
add_subdirectory(test)
add_subdirectory(example)
add_subdirectory(tool)
add_subdirectory(project)
add_subdirectory(src/lit)

add_executable(pscm_c src/c/main.c)
add_executable(pscm_cc)
add_subdirectory(src/c)
target_link_libraries(pscm_cc PRIVATE pscm)
