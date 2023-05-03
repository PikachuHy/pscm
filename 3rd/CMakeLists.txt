add_subdirectory(doctest)
add_subdirectory(spdlog)
# add_subdirectory(glob)

add_library(glob
  glob/include/glob/glob.h
  glob/source/glob.cpp
)
target_include_directories(glob PUBLIC glob/include)
