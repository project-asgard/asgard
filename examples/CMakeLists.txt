cmake_minimum_required(VERSION 3.19)

project(asgard_examples VERSION @asgard_VERSION_MAJOR@.@asgard_VERSION_MINOR@.@asgard_VERSION_PATCH@ LANGUAGES CXX)

find_package(asgard @asgard_VERSION_MAJOR@.@asgard_VERSION_MINOR@.@asgard_VERSION_PATCH@ REQUIRED
             PATHS "@CMAKE_INSTALL_PREFIX@")

add_executable(example_continuity_2d  continuity_2d.cpp)

target_link_libraries(example_continuity_2d asgard::asgard)

if (asgard_PYTHON_FOUND)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/example_continuity_2d.py"  "${CMAKE_CURRENT_BINARY_DIR}/example_continuity_2d.py")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/example_continuity_2d.m"   "${CMAKE_CURRENT_BINARY_DIR}/example_continuity_2d.m")
endif()
