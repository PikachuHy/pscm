add_subdirectory(doctest)
add_subdirectory(spdlog)
# add_subdirectory(glob)

add_library(glob
  glob/include/glob/glob.h
  glob/source/glob.cpp
)
if (PSCM_USE_CXX20_MODULES)
    target_sources(glob PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            glob.cppm
    )
    add_library(std)
    target_sources(std PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            std.cppm
    )

    add_library(fmt)
    target_sources(fmt PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            fmt.cppm
    )
    target_include_directories(fmt PUBLIC fmt/include)
    target_include_directories(fmt PRIVATE fmt/src)
    target_link_libraries(fmt PUBLIC std)
    add_library(subprocess)
    target_sources(subprocess PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            subprocess.cppm
    )
    file(GLOB SUBPROCESS_SRCS "subprocess/src/cpp/subprocess/*.cpp")
    target_sources(subprocess PRIVATE ${SUBPROCESS_SRCS})
    target_include_directories(subprocess PUBLIC subprocess/src/cpp)
    add_library(json)
    target_sources(json PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            json.cppm
    )
    target_include_directories(json PRIVATE json/include)
endif ()
target_include_directories(glob PUBLIC glob/include)
if (${CMAKE_CXX_STANDARD} LESS 17)
    target_compile_definitions(glob PUBLIC GLOB_USE_GHC_FILESYSTEM)
    target_include_directories(glob PRIVATE filesystem/include)
endif ()