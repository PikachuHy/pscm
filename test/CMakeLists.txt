enable_testing()
function(add_pscm_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE doctest::doctest pscm)
    add_test(NAME ${name} COMMAND ${name})
endfunction()
FILE(GLOB tests_files "*_tests.cpp")
foreach (tests_file ${tests_files})
    # message(STATUS "tests_file: ${tests_file}")
    get_filename_component(tests_filename ${tests_file} NAME_WE)
    # message(STATUS "name: ${tests_filename}")
    string(LENGTH ${tests_filename} name_len)
    math(EXPR name_len "${name_len}-1")
    string(SUBSTRING ${tests_filename} 0 ${name_len} test_case_name)
    # message(STATUS "test case: ${test_case_name}")
    add_pscm_test(${test_case_name} ${tests_file})
endforeach ()

add_pscm_test(sicp_ch1_tests sicp/ch1_tests.cpp)
add_pscm_test(r4rs_test r4rs/r4rs_tests.cpp)
add_pscm_test(module_test module/load_path_tests.cpp)
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
add_test(
        NAME module_r5rs_test.scm_DIRECT
        COMMAND $<TARGET_FILE:pscm_main> -m DIRECT -s r5rs_test.scm
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)
add_test(
        NAME module_texmacs_module_DIRECT
        COMMAND $<TARGET_FILE:pscm_main> -m DIRECT -s init.scm
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)
endif()