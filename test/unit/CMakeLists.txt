# Collect pscm sources (same as src/c/ but without main.cc)
file(GLOB pscm_sources "${PSCM_SOURCE_DIR}/src/c/*.cc")
list(REMOVE_ITEM pscm_sources "${PSCM_SOURCE_DIR}/src/c/main.cc")

add_executable(cont_base_mismatch_test cont_base_mismatch_test.cc ${pscm_sources})
target_include_directories(cont_base_mismatch_test PRIVATE "${PSCM_SOURCE_DIR}/src/c")
target_compile_options(cont_base_mismatch_test PRIVATE ${CMAKE_CXX_FLAGS})

add_test(NAME cont_base_mismatch_test COMMAND cont_base_mismatch_test)
