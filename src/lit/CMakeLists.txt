find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_property(PSCM_LIT_CONFIG_FILES GLOBAL PROPERTY PSCM_LIT_CONFIG_FILES)
list(LENGTH PSCM_LIT_CONFIG_FILES file_len)
math(EXPR file_last "${file_len} - 1")

include(lit)

set(LIT_BASE_DIR ${PSCM_BINARY_DIR}/bin)

make_paths_relative(
    PSCM_LIT_CONFIG_FILES "${LIT_BASE_DIR}" "${PSCM_LIT_CONFIG_FILES}")
  make_paths_relative(
    LLVM_SOURCE_DIR "${LIT_BASE_DIR}" "${PSCM_SOURCE_DIR}")

set(PSCM_LIT_CONFIG_MAP "${PSCM_LIT_PATH_FUNCTION}\n")
if (${file_last} GREATER -1)
  foreach(i RANGE 0 ${file_last} 2)
    list(GET PSCM_LIT_CONFIG_FILES ${i} main_config)
    math(EXPR i1 "${i} + 1")
    list(GET PSCM_LIT_CONFIG_FILES ${i1} site_out)
    set(map "map_config(path(r'${main_config}'), path(r'${site_out}'))")
    set(PSCM_LIT_CONFIG_MAP "${PSCM_LIT_CONFIG_MAP}\n${map}")
  endforeach()
endif()


configure_file(
    pscm-lit.in
    ${PROJECT_BINARY_DIR}/bin/pscm-lit
)
