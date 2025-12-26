enable_testing()
add_subdirectory(cont)
add_subdirectory(core)
add_subdirectory(base)
add_subdirectory(sicp)
add_subdirectory(macro)


add_test(
    NAME load_path_tests.scm
    COMMAND $<TARGET_FILE:pscm_cc> --test load_path_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME basic_module_tests
    COMMAND $<TARGET_FILE:pscm_cc> --test basic_module_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME module_export_tests
    COMMAND $<TARGET_FILE:pscm_cc> --test module_export_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME module_re_export_tests
    COMMAND $<TARGET_FILE:pscm_cc> --test module_re_export_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)


add_test(
    NAME module_isolation_tests
    COMMAND $<TARGET_FILE:pscm_cc> --test module_isolation_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME module_lookup_tests
    COMMAND $<TARGET_FILE:pscm_cc> --test module_lookup_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
        NAME module_r5rs_test.scm_DIRECT
        COMMAND $<TARGET_FILE:pscm_cc> --test r5rs_test.scm
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME module_public_interface_tests
    COMMAND $<TARGET_FILE:pscm_cc> --test module_public_interface_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME module_map_tests
    COMMAND $<TARGET_FILE:pscm_cc> --test module_map_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME module_use_tests
    COMMAND $<TARGET_FILE:pscm_cc> --test module_use_tests.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)
