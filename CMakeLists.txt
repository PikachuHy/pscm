cmake_minimum_required(VERSION 3.26)
project(pscm VERSION 0.5.0 LANGUAGES C CXX)
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PSCM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PSCM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_executable(pscm_cc)
add_subdirectory(src/c)
add_subdirectory(src/lit)
add_subdirectory(test)
