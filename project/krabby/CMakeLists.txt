set(CMAKE_CXX_STANDARD 20)
add_executable(krabby src/main.cpp)
if (EMSCRIPTEN)
    add_custom_command(
            TARGET krabby PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/asset
            ${CMAKE_BINARY_DIR}/asset)
    add_custom_command(
            TARGET krabby PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/images
            ${CMAKE_CURRENT_BINARY_DIR}/images)
    add_custom_command(
            TARGET krabby PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/icon
            ${CMAKE_CURRENT_BINARY_DIR}/icon)
    add_custom_command(
            TARGET krabby POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/krabby.html
            ${CMAKE_CURRENT_BINARY_DIR}/index.html)
    target_compile_options(krabby PUBLIC "-sUSE_SDL=2")
    target_compile_options(krabby PUBLIC "-sUSE_SDL_TTF=2")
    target_link_options(krabby PUBLIC "-sUSE_SDL=2")
    target_link_options(krabby PUBLIC "-sUSE_SDL_TTF=2")
    target_link_options(krabby PUBLIC --preload-file asset)
    # do not scan C++20 Modules deps
    set_target_properties(krabby PROPERTIES
            CXX_SCAN_FOR_MODULES OFF
    )
else ()
    add_custom_command(
            TARGET krabby PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/asset
            ${CMAKE_CURRENT_BINARY_DIR}/asset)
    find_package(SDL2 REQUIRED)
    find_package(SDL2_ttf REQUIRED)
    find_package(SDL2_image REQUIRED)
    find_package(SDL2_mixer REQUIRED)
    target_link_libraries(krabby PRIVATE SDL2::SDL2)
    target_link_libraries(krabby PRIVATE SDL2_ttf::SDL2_ttf)
    target_link_libraries(krabby PRIVATE SDL2_image::SDL2_image)
    target_link_libraries(krabby PRIVATE SDL2_mixer::SDL2_mixer)
endif ()
