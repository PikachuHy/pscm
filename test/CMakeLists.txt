enable_testing()
function(add_pscm_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE doctest::doctest pscm)
    add_test(NAME ${name} COMMAND ${name})
endfunction()
add_pscm_test(sicp_ch1_tests sicp/ch1_tests.cpp)
add_pscm_test(macro_test macro_tests.cpp)
add_pscm_test(number_test number_tests.cpp)
add_pscm_test(math_test math_tests.cpp)
add_pscm_test(parser_test parser_tests.cpp)
add_pscm_test(cell_test cell_tests.cpp)
add_pscm_test(callcc_test callcc_tests.cpp)
add_pscm_test(for_each_test for_each_tests.cpp)
add_pscm_test(map_test map_tests.cpp)
add_pscm_test(cond_test cond_tests.cpp)
add_pscm_test(do_test do_tests.cpp)
add_pscm_test(list_test list_tests.cpp)
add_pscm_test(char_test char_tests.cpp)
add_pscm_test(apply_test apply_tests.cpp)
add_pscm_test(port_test port_tests.cpp)
add_pscm_test(let_test let_tests.cpp)
add_pscm_test(letrec_test letrec_tests.cpp)
add_pscm_test(append_test append_tests.cpp)
add_pscm_test(r4rs_test r4rs/r4rs_tests.cpp)
if (EMSCRIPTEN)
elseif (WIN32)
# TODO: handle EOF
else()
add_test(
    NAME r4rstest.scm_DIRECT
    COMMAND $<TARGET_FILE:pscm_main> -m DIRECT -s r4rstest.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)
add_test(
    NAME r4rstest.scm_REGISTER_MACHINE
    COMMAND $<TARGET_FILE:pscm_main> -m REGISTER_MACHINE -s r4rstest.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)
add_test(
    NAME r4rs_cont_test.scm_REGISTER_MACHINE
    COMMAND $<TARGET_FILE:pscm_main> -m REGISTER_MACHINE -s r4rs_cont_test.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)
add_test(
    NAME r4rs_load.scm_DIRECT
    COMMAND $<TARGET_FILE:pscm_main> -m DIRECT -s load.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)
add_test(
    NAME r5rstest.scm_DIRECT
    COMMAND $<TARGET_FILE:pscm_main> -m DIRECT -s r5rstest.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r5rs
)
add_test(
    NAME r5rstest.scm_REGISTER_MACHINE
    COMMAND $<TARGET_FILE:pscm_main> -m REGISTER_MACHINE -s r5rstest.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r5rs
)
add_test(
    NAME r5rs_load.scm_DIRECT
    COMMAND $<TARGET_FILE:pscm_main> -m DIRECT -s load.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r5rs
)
add_test(
    NAME r5rs_load.scm_REGISTER_MACHINE
    COMMAND $<TARGET_FILE:pscm_main> -m REGISTER_MACHINE -s load.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r5rs
)
endif()