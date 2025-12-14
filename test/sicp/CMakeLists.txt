include(lit)

pscm_configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

pscm_add_lit_testsuite(check-sicp "Running the base regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
    pscm_cc
)
