cmake_minimum_required(VERSION 3.19)

project(asgard_examples VERSION @asgard_VERSION_MAJOR@.@asgard_VERSION_MINOR@.@asgard_VERSION_PATCH@ LANGUAGES CXX)

find_package(asgard @asgard_VERSION_MAJOR@.@asgard_VERSION_MINOR@.@asgard_VERSION_PATCH@ REQUIRED
             PATHS "@__asgard_install_prefix@")

add_executable(continuity_2d  continuity_2d.cpp)
add_executable(inputs_1d      inputs_1d.cpp)

target_link_libraries(continuity_2d  asgard::asgard)
target_link_libraries(inputs_1d      asgard::asgard)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_1.txt"   "${CMAKE_CURRENT_BINARY_DIR}/inputs_1d_1.txt" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_2.txt"   "${CMAKE_CURRENT_BINARY_DIR}/inputs_1d_2.txt" COPYONLY)

if (asgard_PYTHON_FOUND)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/continuity_2d.py"  "${CMAKE_CURRENT_BINARY_DIR}/continuity_2d.py" COPYONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/continuity_2d.m"   "${CMAKE_CURRENT_BINARY_DIR}/continuity_2d.m" COPYONLY)

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d.py"   "${CMAKE_CURRENT_BINARY_DIR}/inputs_1d.py" COPYONLY)
endif()
