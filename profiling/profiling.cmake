
# todo:
# - make graphviz build optional
if (ASGARD_PROFILE_GPROF)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options ("-pg")
    add_link_options ("-pg")
    message (
      "\n"
      "   gprof enabled. For using gprof:\n"
      "   1) build the code with -pg option (will be done for you during 'make')\n"
      "   2) run the executable to be profiled.\n"
      "      this produces a 'gmon.out' in the current directory\n"
      "   3) gprof asgard gmon.out > [output-file.txt]\n"
      "\n"
      "   for more details, see gprof's documentation: https://bit.ly/2UZxdkz\n"
      "\n"
      "\n"
      "   To get graphical results try graphviz + gprof2dot:\n"
      "   gprof asgard gmon.out \\ \n"
      "   | ../contrib/gprof2dot/bin/gprof2dot.py -n 2 -e 1 \\ \n"
      "   | ../contrib/graphviz/bin/dot -Tpdf -o profile.pdf\n"
      "\n"
    )

  # build graphviz if needed
  # TODO detect if it is on the system
  set (graphviz_PATH ${CMAKE_SOURCE_DIR}/contrib/graphviz)
  if (NOT EXISTS ${graphviz_PATH}/bin/dot)
    message (STATUS "graphviz not found. building from source")
    include (ExternalProject)
    ExternalProject_Add (graphviz-ext
      PREFIX contrib/graphviz
      URL https://graphviz.gitlab.io/pub/graphviz/stable/SOURCES/graphviz.tar.gz
      DOWNLOAD_NO_PROGRESS 1
      BUILD_IN_SOURCE 1
      USES_TERMINAL_CONFIGURE 1
      CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/graphviz/src/graphviz-ext/configure --prefix=${graphviz_PATH}
      BUILD_COMMAND make
      INSTALL_COMMAND make install
    )
  else ()
    message (STATUS "using graphviz found at ${graphviz_PATH}")
  endif ()

  # grab gprof2dot (we don't store it in the repo or distribute)
  set (gprof2dot_PATH ${CMAKE_SOURCE_DIR}/contrib/gprof2dot)
  if (NOT EXISTS ${gprof2dot_PATH}/bin/gprof2dot.py)
    message (STATUS "gprof2dot not found. downloading")
    include (ExternalProject)
    ExternalProject_Add (gprof2dot-ext
      PREFIX contrib/gprof2dot
      URL https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py
      DOWNLOAD_NO_PROGRESS 1
      DOWNLOAD_NO_EXTRACT 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND mkdir -p ${gprof2dot_PATH}/bin
      COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/contrib/gprof2dot/src/gprof2dot.py ${gprof2dot_PATH}/bin/
      COMMAND chmod 700 ${gprof2dot_PATH}/bin/gprof2dot.py
    )
  else ()
    message (STATUS "using gprof2dot found at ${gprof2dot_PATH}")
  endif ()

  else ()
    message (FATAL_ERROR "must use GCC to enable gprof suppport")
  endif ()
endif ()

if (ASGARD_PROFILE_XRAY)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options ("")
  else ()
    message (FATAL_ERROR "must use clang to enable xray suppport")
  endif ()
endif ()

if (ASGARD_PROFILE_GPERF)

endif ()

if (ASGARD_PROFILE_VALGRIND)

endif ()
