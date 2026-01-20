cmake_minimum_required(VERSION 3.19)

project(asgard_examples VERSION @asgard_VERSION@ LANGUAGES CXX)

find_package(asgard @asgard_VERSION@ REQUIRED PATHS "@__asgard_install_prefix@")

# additional examples following the same pattern
foreach (_example @_asgard_doc_pdes_str@)
    add_executable (${_example}  ${_example}.cpp)
    target_link_libraries (${_example}  asgard::asgard)
endforeach()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_1.txt"   "${CMAKE_CURRENT_BINARY_DIR}/inputs_1d_1.txt" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d_2.txt"   "${CMAKE_CURRENT_BINARY_DIR}/inputs_1d_2.txt" COPYONLY)

if (asgard_PYTHON_FOUND)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/continuity_2d.py"  "${CMAKE_CURRENT_BINARY_DIR}/continuity_2d.py" COPYONLY)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/continuity_2d.m"   "${CMAKE_CURRENT_BINARY_DIR}/continuity_2d.m" COPYONLY)

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/inputs_1d.py"   "${CMAKE_CURRENT_BINARY_DIR}/inputs_1d.py" COPYONLY)

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/slideshow.py"   "${CMAKE_CURRENT_BINARY_DIR}/slideshow.py" COPYONLY)
endif()

# the GPU examples need the proper programming language set (CUDA or HIP)
# this code will not be used, unless CUDA or ROCM have been enabled
if (@__asgard_GPU_COMPONENT@)
    # if using CUDA/HIP, the enable_language(CUDA) or enable_language(HIP)
    # has already been called by the find_package() directive
    # however, the source files have to be marked as using CUDA/HIP

    foreach (_example @_asgard_gpu_pdes_str@)
        add_executable (${_example}  ${_example}.cpp)
        target_link_libraries (${_example}  asgard::asgard)

        # CMake can infer the language of a file in several different ways
        # the most sure way is to set the language explicitly
        set_source_files_properties (${CMAKE_CURRENT_SOURCE_DIR}/${_example}.cpp
                                     PROPERTIES LANGUAGE @__asgard_GPU_LANG@)
    endforeach()
endif()
