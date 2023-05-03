add_executable(pscm-build main.cpp)
target_link_libraries(pscm-build PRIVATE pscm)
target_link_libraries(pscm-build PRIVATE glob)
