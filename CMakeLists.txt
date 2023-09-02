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
        set(CMAKE_CXX_STANDARD 20)
    endif ()
endif ()


set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(pscm)
if (EMSCRIPTEN)
    target_link_options(pscm PUBLIC "-sUSE_ICU=1")
    target_compile_options(pscm PUBLIC "-sUSE_ICU=1")
else ()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules (ICU REQUIRED icu-i18n icu-uc icu-io IMPORTED_TARGET)
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
        src/misc/ICUCompat.cpp
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