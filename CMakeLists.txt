cmake_minimum_required(VERSION 3.26)
project(pscm VERSION 0.3.0 LANGUAGES CXX)
option(PSCM_USE_CXX20_MODULES "enable c++20 modules" OFF)
if (WIN32)
    set(CMAKE_CXX_STANDARD 20)
else ()
    if (PSCM_USE_CXX20_MODULES)
        if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")
            include(cxx_modules_rules_clang.cmake)
            set(CMAKE_CXX_STANDARD 20)
            set(CMAKE_CXX_EXTENSIONS OFF)
        else ()
            message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} not support C++20 Modules now")
        endif ()
    else ()
        set(CMAKE_CXX_STANDARD 14)
    endif ()
endif ()


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# https://cmake.org/cmake/help/latest/module/FindICU.html#module:FindICU
# find_package(ICU REQUIRED) is broken on MacOS due to Failed to find all ICU components
# TODO: better way to handle icu4c
if (NOT DEFINED ICU4C_ROOT)
  message(FATAL_ERROR "use cmake -DICU4C_ROOT=/path/to/your/icu4c")
endif()
function(_import_icu_ component)
if(NOT TARGET ICU::${component})
add_library(ICU::${component} UNKNOWN IMPORTED)
find_library(ICU_${component}_LIBRARY NAMES icu${component} HINTS ${ICU4C_ROOT}/lib)
if (ICU_${component}_LIBRARY)
message(STATUS "ICU_${component}_LIBRARY: ${ICU_${component}_LIBRARY}")
set_target_properties(ICU::${component} PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${ICU4C_ROOT}/include"
          INTERFACE_LINK_DIRECTORIES "${ICU4C_ROOT}/lib"
          INTERFACE_LINK_LIBRARIES ${ICU_${component}_LIBRARY}
          IMPORTED_LOCATION "${ICU_${component}_LIBRARY}"
          )
          message(INFO "Imported ICU::${component}")
else()
  message(WARNING "icu${component} not found")
endif()
endif()
endfunction(_import_icu_)
_import_icu_(uc)
_import_icu_(i18n)
_import_icu_(io)
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
add_library(pscm)
target_link_libraries(pscm PUBLIC ICU::uc ICU::i18n ICU::io)
target_include_directories(pscm PUBLIC include)
target_include_directories(pscm PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generate)
target_include_directories(pscm PUBLIC 3rd/UniversalStacktrace)
target_include_directories(pscm PUBLIC 3rd/cpp-linenoise)
if (USE_CPP_STD_COMPAT)
    target_include_directories(pscm PUBLIC 3rd/variant/include)
    target_include_directories(pscm PUBLIC 3rd/optional/include)
    target_include_directories(pscm PUBLIC 3rd/filesystem/include)
    target_include_directories(pscm PUBLIC 3rd/string-view-lite/include)
endif ()

file(GLOB PSCM_SRCS "src/*.cpp")
target_sources(pscm PRIVATE ${PSCM_SRCS})
target_sources(pscm PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generate/src/version.cpp
        src/logger/Logger.cpp
        src/logger/Appender.cpp
        src/misc/SourceLocation.cpp
)
if (PSCM_USE_CXX20_MODULES)
    target_compile_definitions(pscm PUBLIC PSCM_USE_CXX20_MODULES)
    target_sources(pscm PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            src/logger/logger.cppm
            src/misc/misc.cppm
            src/pscm.cppm
            src/compat.cppm
            3rd/linenoise.cppm
    )
    target_link_libraries(pscm PUBLIC fmt)
else ()
    target_link_libraries(pscm PUBLIC spdlog::spdlog)
endif ()
if (EMSCRIPTEN)
    target_compile_definitions(pscm PUBLIC WASM_PLATFORM)
    target_link_options(pscm PUBLIC "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap")
    target_link_options(pscm PUBLIC "-sSTACK_SIZE=6553600")
    target_link_options(pscm PUBLIC "-sNO_DISABLE_EXCEPTION_CATCHING")
    add_subdirectory(wasm)
else ()
    add_executable(pscm_main main.cpp)
    target_link_libraries(pscm_main PRIVATE pscm)
endif ()
add_subdirectory(test)
add_subdirectory(example)
add_subdirectory(tool)
add_subdirectory(project)