find_package(Doxygen REQUIRED OPTIONAL_COMPONENTS dot)

set(DOXYGEN_GENERATE_HTML      "YES")
set(DOXYGEN_DISABLE_INDEX       "NO")
set(DOXYGEN_GENERATE_TREEVIEW  "YES") # left-hand index
set(DOXYGEN_SORT_GROUP_NAMES    "NO")
set(DOXYGEN_SORT_BRIEF_DOCS     "NO")
set(DOXYGEN_SORT_MEMBER_DOCS    "NO")
set(DOXYGEN_COLLABORATION_GRAPH "NO")
set(DOXYGEN_INCLUDE_GRAPH       "NO")
set(DOXYGEN_INCLUDED_BY_GRAPH   "NO")
set(DOXYGEN_GRAPHICAL_HIERARCHY "NO")
set(DOXYGEN_CLASS_GRAPH         "NO")
set(DOXYGEN_CLASS_DIAGRAMS      "NO")

set(DOXYGEN_HTML_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../html/")

set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
set(DOXYGEN_HTML_EXTRA_STYLESHEET  "${CMAKE_CURRENT_SOURCE_DIR}/asgard.css")
set(DOXYGEN_HTML_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/header.html")
set(DOXYGEN_HTML_FOOTER "${CMAKE_CURRENT_SOURCE_DIR}/footer.html")

set(DOXYGEN_HTML_COLORSTYLE_HUE    "146")
set(DOXYGEN_HTML_COLORSTYLE_SAT    "100")
set(DOXYGEN_HTML_COLORSTYLE_GAMMA  "190")

set(DOXYGEN_PROJECT_NAME "ASGarD - Adaptive Sparse Grid Discretization v${asgard_VERSION_MAJOR}.${asgard_VERSION_MINOR}.${asgard_VERSION_PATCH}")
set(DOXYGEN_PROJECT_NUMBER "")

if (NOT DOXYGEN_INTERNAL_DOCS)
# the \internal command works for functions and namespaces, but not classes
# have to exclude the classes manually
    set(DOXYGEN_EXCLUDE_SYMBOLS asgard::vecstrview asgard::split_views asgard::matrix_factor)

    set(DOXYGEN_PREDEFINED "__ASGARD_DOXYGEN_SKIP;__ASGARD_DOXYGEN_SKIP_INTERNAL;")
endif()

doxygen_add_docs(asgard_doxygen
                 README.md
                 doxygen/installation.md
                 doxygen/basic_usage.md
                 doxygen/plotting.md
                 src/asgard.hpp
                 src/asgard_discretization.hpp
                 src/asgard_program_options.hpp
                 src/asgard_reconstruct.hpp
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
                 COMMENT "Building the ${PROJECT_NAME} documentation")

# asgard_doxygen is not automatically added to "all"
# must add a dependence to ensure it will build
add_dependencies(asgard_exe asgard_doxygen)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../html DESTINATION "share/asgard/docs")
