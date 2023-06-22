add_subdirectory(doctest)
add_subdirectory(spdlog)
# add_subdirectory(glob)

add_library(glob
  glob/include/glob/glob.h
  glob/source/glob.cpp
)
target_include_directories(glob PUBLIC glob/include)
if (${CMAKE_CXX_STANDARD} LESS 17)
  target_compile_definitions(glob PUBLIC GLOB_USE_GHC_FILESYSTEM)
  target_include_directories(glob PRIVATE filesystem/include)
endif ()