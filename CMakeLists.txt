cmake_minimum_required(VERSION 3.22)
project(pscm VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
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
configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/pscm/version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/generate/pscm/version.h
        @ONLY
)
add_subdirectory(3rd)
add_library(pscm)
target_include_directories(pscm PUBLIC include)
target_include_directories(pscm PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/generate)
aux_source_directory(src PSCM_SRCS)
target_sources(pscm PRIVATE ${PSCM_SRCS})
target_link_libraries(pscm PUBLIC spdlog::spdlog)
if (EMSCRIPTEN)
    target_compile_definitions(pscm PUBLIC WASM_PLATFORM)
    target_link_options(pscm PUBLIC "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap")
    target_link_options(pscm PUBLIC "-sNO_DISABLE_EXCEPTION_CATCHING")
    add_subdirectory(wasm)
else ()
    add_executable(pscm_main main.cpp)
    target_link_libraries(pscm_main PRIVATE pscm)
endif ()
add_subdirectory(test)
add_subdirectory(example)
add_subdirectory(tool)
