add_executable(pscm_wasm pscm.cpp)
set_target_properties(pscm_wasm PROPERTIES OUTPUT_NAME pscm)
target_link_libraries(pscm_wasm PRIVATE pscm)
