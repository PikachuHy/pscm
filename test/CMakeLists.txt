enable_testing()
add_subdirectory(unit)
add_subdirectory(cont)
add_subdirectory(core)
add_subdirectory(base)
add_subdirectory(sicp)
add_subdirectory(macro)
add_subdirectory(gc)
add_subdirectory(c_api)

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
    NAME r4rs_mini.scm
    COMMAND $<TARGET_FILE:pscm_cc> --test r4rs_mini.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)

add_test(
    NAME r4rs_mini_2.scm
    COMMAND $<TARGET_FILE:pscm_cc> --test r4rs_mini_2.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)

add_test(
    NAME r4rstest.scm
    COMMAND $<TARGET_FILE:pscm_cc> --test r4rstest.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)

add_test(
    NAME r4rs_load_letrec.scm
    COMMAND $<TARGET_FILE:pscm_cc> --test r4rs_load_letrec.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)

add_test(
    NAME r4rs_load
    COMMAND $<TARGET_FILE:pscm_cc> --test load.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)

add_test(
    NAME r4rs_cont_test.scm
    COMMAND $<TARGET_FILE:pscm_cc> --test r4rs_cont_test.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r4rs
)


add_test(
    NAME r5rstest.scm
    COMMAND $<TARGET_FILE:pscm_cc> --test r5rstest.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r5rs
)

add_test(
    NAME r5rs_cont_test.scm
    COMMAND $<TARGET_FILE:pscm_cc> --test r5rs_cont_test.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r5rs
)

add_test(
        NAME r5rs_load
        COMMAND $<TARGET_FILE:pscm_cc> --test load.scm
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/r5rs
)

add_test(
        NAME module_r5rs_test.scm
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

add_test(
    NAME public_interface_isolation
    COMMAND $<TARGET_FILE:pscm_cc> --test public_interface_isolation.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME binder_custom
    COMMAND $<TARGET_FILE:pscm_cc> --test binder_custom.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME autoload_lazy
    COMMAND $<TARGET_FILE:pscm_cc> --test autoload_lazy.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME define_module_options
    COMMAND $<TARGET_FILE:pscm_cc> --test define_module_options.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME module_private_macro
    COMMAND $<TARGET_FILE:pscm_cc> --test module_private_macro.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module
)

add_test(
    NAME module_texmacs_module
    COMMAND $<TARGET_FILE:pscm_cc> --test init.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME nested_module
    COMMAND $<TARGET_FILE:pscm_cc> --test nested_module.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME re_export_conflict
    COMMAND $<TARGET_FILE:pscm_cc> --test re_export_conflict.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME module_use_binding
    COMMAND $<TARGET_FILE:pscm_cc> --test module_use_binding.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME string_ports
    COMMAND $<TARGET_FILE:pscm_cc> --test string_ports.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME file_ports
    COMMAND $<TARGET_FILE:pscm_cc> --test file_ports.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME cont_module
    COMMAND $<TARGET_FILE:pscm_cc> --test cont_module.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME dynwind_module
    COMMAND $<TARGET_FILE:pscm_cc> --test dynwind_module.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME texmacs_macros
    COMMAND $<TARGET_FILE:pscm_cc> --test texmacs_macros.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME define_public_macro
    COMMAND $<TARGET_FILE:pscm_cc> --test define_public_macro.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

# Incremental TeXmacs init loading test (expected to fail initially)
add_test(
    NAME texmacs_init_loading
    COMMAND $<TARGET_FILE:pscm_cc> --test load_texmacs_init.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)
# Note: No PASS_REGULAR_EXPRESSION — this test won't pass until the full init loads

add_test(
    NAME float_cmp
    COMMAND $<TARGET_FILE:pscm_cc> --test float_cmp.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)

add_test(
    NAME macro_expr_pos
    COMMAND $<TARGET_FILE:pscm_cc> --test macro_expr_pos.scm
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/module/texmacs
)