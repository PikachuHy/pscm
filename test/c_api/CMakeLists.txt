# Collect pscm sources (same as src/c/ but without main.cc)
file(GLOB pscm_sources "${PSCM_SOURCE_DIR}/src/c/*.cc")
list(REMOVE_ITEM pscm_sources "${PSCM_SOURCE_DIR}/src/c/main.cc")
list(REMOVE_ITEM pscm_sources "${PSCM_SOURCE_DIR}/src/c/repl.cc")

add_executable(c_api_basic c_api_basic.cpp ${pscm_sources})
target_include_directories(c_api_basic PRIVATE "${PSCM_SOURCE_DIR}/src/c")
target_compile_options(c_api_basic PRIVATE ${CMAKE_CXX_FLAGS})

add_executable(c_api_module c_api_module.cpp ${pscm_sources})
target_include_directories(c_api_module PRIVATE "${PSCM_SOURCE_DIR}/src/c")
target_compile_options(c_api_module PRIVATE ${CMAKE_CXX_FLAGS})

add_test(NAME c_api_module COMMAND c_api_module)

add_test(NAME c_api_basic COMMAND c_api_basic)

add_executable(c_api_error c_api_error.cpp ${pscm_sources})
target_include_directories(c_api_error PRIVATE "${PSCM_SOURCE_DIR}/src/c")
target_compile_options(c_api_error PRIVATE ${CMAKE_CXX_FLAGS})

add_test(NAME c_api_error COMMAND c_api_error)
