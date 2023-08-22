add_executable(pscm-build
        Rule.cpp
        DepsScanner.cpp
        Action.cpp
        Label.cpp
)
file(GLOB PSCM_BUILD_SRCS "*.cppm")
target_sources(pscm-build PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ${PSCM_BUILD_SRCS}
        main.cpp
)
target_link_libraries(pscm-build PRIVATE pscm fmt)
target_link_libraries(pscm-build PRIVATE glob)
target_link_libraries(pscm-build PRIVATE subprocess)
target_link_libraries(pscm-build PRIVATE json)
