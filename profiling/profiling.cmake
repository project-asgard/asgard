
###############################################################################
## Profiling support
###############################################################################

set (PROFILE_DEPS "")
set (PROFILE_LIBS "")

###############################################################################
## Profiling dependencies
#
# These things are needed by more than one profiling method handled below, so
# extract this functionality into functions
#
# FIXME
#  - make graphviz build optional
###############################################################################

function (get_graphviz)
if (NOT ASGARD_BUILD_PROFILE_DEPS)
  # search for graphviz's dot under user-supplied path(s)
  if (ASGARD_GRAPHVIZ_PATH)
    find_program (GRAPHVIZ_DOT_PATH dot PATHS ${ASGARD_GRAPHVIZ_PATH}
      PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (GRAPHVIZ_DOT_PATH)
      set (graphviz_PATH ${GRAPHVIZ_DOT_PATH})
    endif ()
  endif ()

  # search for graphviz's dot in some typical locations
  if (NOT graphviz_PATH)
    find_program (GRAPHVIZ_DOT_PATH dot PATHS /usr/ /usr/local/
      PATH_SUFFIXES bin NO_DEFAULT_PATH)
    if (GRAPHVIZ_DOT_PATH)
      set (graphviz_PATH ${GRAPHVIZ_DOT_PATH})
    endif ()
  endif ()

  if (graphviz_PATH)
    message (STATUS "using external graphviz found at ${graphviz_PATH}")
  endif ()
endif ()

# if cmake couldn't find other blas/lapack, or the user asked to build openblas
if (NOT graphviz_PATH)
  # build graphviz if needed
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
      BUILD_COMMAND make -j
      INSTALL_COMMAND make install
    )
  else ()
    message (STATUS "using contrib graphviz found at ${graphviz_PATH}")
  endif ()
endif ()
endfunction ()

###############################################################################
## GNU gprof
#
# FIXME
#  - make gprof2dot build optional
#
#  if cmake needs to "build graphviz", and it remains from a previous build,
#  cmake will skip the build and reuse it
###############################################################################
if (ASGARD_PROFILE_GPROF)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options ("-pg")
    add_link_options ("-pg")
    message (
      "\n"
      "   gprof enabled. For using gprof:\n"
      "   1) build/link the code with -pg option (will be done for you during 'make')\n"
      "   2) run the executable to be profiled. e.g.\n"
      "      $ ./asgard -p continuity_6 -l 8 -d 3\n"
      "      this produces a 'gmon.out' in the current directory\n"
      "   3) analyze the restuls to get the timings and call graph\n"
      "      $ gprof asgard gmon.out > [output-file.txt]\n"
      "\n"
      "   for more details, see gprof's documentation: https://bit.ly/2UZxdkz\n"
      "\n"
      "\n"
      "   To get graphical results try graphviz + gprof2dot:\n"
      "   gprof asgard gmon.out \\ \n"
      "   | ../contrib/gprof2dot/bin/gprof2dot.py -n 2 -e 1 -w \\ \n"
      "   | ../contrib/graphviz/bin/dot -Tpdf -o profile.pdf\n"
      "\n"
    )

    # find graphviz, build if needed
    get_graphviz()

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

###############################################################################
## LLVM XRAY
#
# FIXME
#  - make flamegraph build optional
###############################################################################

if (ASGARD_PROFILE_XRAY)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options ("-fxray-instrument")
    add_link_options ("-fxray-instrument")
    message (
      "\n"
      "   LLVM XRAY enabled. to use:\n"
      "   1) build/link the code with -fxray-instrument option (done for you during 'make')\n"
      "   2) run the executable to be profiled with:\n"
      "      $ XRAY_OPTIONS=\"patch_premain=true xray_mode=xray-basic verbosity=1\" ./asgard -p continuity_6 -l 8 -d 3\n"
      "      this produces a uniquely-hashed 'xray-log.asgard.[hash]' file\n"
      "   3) analyze the results with\n"
      "      $ llvm-xray account xray-log.asgard.[hash] -sort=sum -sortorder=dsc -instr_map ./asgard\n"
    )

    # grab FlameGraph (we don't store it in the repo or distribute)
    set (flamegraph_PATH ${CMAKE_SOURCE_DIR}/contrib/FlameGraph)
    if (NOT EXISTS ${flamegraph_PATH}/flamegraph.pl)
      message (STATUS "flamegraph not found. downloading")
      include (ExternalProject)
      ExternalProject_Add (flamegraph-ext
        SOURCE_DIR ${flamegraph_PATH}
        GIT_REPOSITORY https://github.com/brendangregg/FlameGraph
        GIT_PROGRESS 1
        GIT_SHALLOW 1
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
      )
    else ()
      message (STATUS "using flamegraph.pl found at ${flamegraph_PATH}")
    endif ()

  else ()
    message (FATAL_ERROR "must use clang to enable xray suppport")
  endif ()
endif ()

###############################################################################
## gperftools (formerly google performance tools)
#
# FIXME allow user to point to previously installed libprofiler, etc.
###############################################################################

if (ASGARD_PROFILE_GPERF)
  #add_link_options ("-lprofiler")
    message (
      "\n"
      "   gperftools enabled. to use:\n"
      "   1) link the code with -lprofiler option (done for you during 'make')\n"
      "   2) run the executable to be profiled with:\n"
      "      $ CPUPROFILE=some-name.prof ./asgard -p continuity_6 -l 8 -d 3\n"
      "      this produces a profile file name 'some-name.prof'\n"
      "   3) analyze the results with\n"
      "      $ pprof --pdf ./asgard some-name.prof > some-other.pdf\n"
    )

  # build gperftools if needed
  set (gperftools_PATH ${CMAKE_SOURCE_DIR}/contrib/gperftools)
  if (NOT EXISTS ${gperftools_PATH}/bin/pprof)
    message (STATUS "gperftools not found. building from source")
    set (PROFILE_DEPS gperftools-ext)
    include (ExternalProject)
    ExternalProject_Add (gperftools-ext
      PREFIX contrib/gperftools
      GIT_REPOSITORY https://github.com/gperftools/gperftools
      GIT_PROGRESS 1
      GIT_SHALLOW 1
      BUILD_IN_SOURCE 1
      USES_TERMINAL_CONFIGURE 1
      CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/gperftools/src/gperftools-ext/autogen.sh
      COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/gperftools/src/gperftools-ext/configure --prefix=${gperftools_PATH}
      BUILD_COMMAND make -j
      INSTALL_COMMAND make install
    )
  else ()
    message (STATUS "using gperftools found at ${gperftools_PATH}")
  endif ()

  # find graphviz, build if needed
  get_graphviz()

  set (PROFILE_LIBS -Wl,-no-as-needed "${gperftools_PATH}/lib/libprofiler.so")
endif ()

###############################################################################
## linux perf
###############################################################################

if (ASGARD_PROFILE_PERF)
  message (FATAL_ERROR "linux perf not supported yet :'(. Please submit an issue or PR.")
endif ()

###############################################################################
## Valgrind
###############################################################################

if (ASGARD_PROFILE_VALGRIND)
  message (FATAL_ERROR "valgrind not supported yet :'(. Please submit an issue or PR.")
endif ()
