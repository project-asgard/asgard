
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
#  - abstract out a "get_*" function that generalizes all of this functionality
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

# if cmake couldn't find other graphviz, or the user asked to build deps
if (NOT graphviz_PATH)
  # build graphviz if needed
  set (graphviz_PATH ${CMAKE_SOURCE_DIR}/contrib/graphviz)
  if (NOT EXISTS ${graphviz_PATH}/bin/dot)
    message (STATUS "graphviz not found. building from source")
    include (ExternalProject)
    ExternalProject_Add (graphviz-ext
      UPDATE_COMMAND ""
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

function (get_gperftools)
  set (gperftools_PATH ${CMAKE_SOURCE_DIR}/contrib/gperftools)
  if (NOT EXISTS ${gperftools_PATH}/lib/libprofiler.so)
    set (PROFILE_DEPS gperftools-ext PARENT_SCOPE) # tell the caller to build
    message (STATUS "gperftools not found. building from source")
    include (ExternalProject)
    ExternalProject_Add (gperftools-ext
      UPDATE_COMMAND ""
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

  # pass this back to the caller
  set (gperftools_PATH ${CMAKE_SOURCE_DIR}/contrib/gperftools PARENT_SCOPE)
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
      "Possibly more explanation about various profiling tools enabled here\n"
      "exists at the ASGarD wiki page on profiling.\n"
    )

    # find graphviz, build if needed
    get_graphviz()

    # grab gprof2dot (we don't store it in the repo or distribute)
    set (gprof2dot_PATH ${CMAKE_SOURCE_DIR}/contrib/gprof2dot)
    if (NOT EXISTS ${gprof2dot_PATH}/bin/gprof2dot.py)
      message (STATUS "gprof2dot not found. downloading")
      include (ExternalProject)
      ExternalProject_Add (gprof2dot-ext
        UPDATE_COMMAND ""
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
      "\n"
      "Possibly more explanation about various profiling tools enabled here\n"
      "exists at the ASGarD wiki page on profiling.\n"
    )

    # grab FlameGraph (we don't store it in the repo or distribute)
    set (flamegraph_PATH ${CMAKE_SOURCE_DIR}/contrib/FlameGraph)
    if (NOT EXISTS ${flamegraph_PATH}/flamegraph.pl)
      message (STATUS "flamegraph not found. downloading")
      include (ExternalProject)
      ExternalProject_Add (flamegraph-ext
        UPDATE_COMMAND ""
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
# CPU profiler
###############################################################################

if (ASGARD_PROFILE_GPERF_CPU)
  message (
    "\n"
    "   gperftools enabled. to use:\n"
    "   1) link the code with -lprofiler option (done for you during 'make')\n"
    "   2) run the executable to be profiled with:\n"
    "      $ CPUPROFILE=some-name.prof ./asgard -p continuity_6 -l 8 -d 3\n"
    "      this produces a profile file name 'some-name.prof'\n"
    "   3) analyze the results with\n"
    "      $ pprof --pdf ./asgard some-name.prof > some-other.pdf\n"
    "\n"
    "Possibly more explanation about various profiling tools enabled here\n"
    "exists at the ASGarD wiki page on profiling.\n"
  )

  # find gperftools, build if needed
  get_gperftools()

  # find graphviz, build if needed
  get_graphviz()

  set (PROFILE_LIBS -Wl,-no-as-needed "${gperftools_PATH}/lib/libprofiler.so")
endif ()


###############################################################################
## gperftools for Memory Allocation (formerly google performance tools)
#
#  Heap profiler
###############################################################################
if (ASGARD_PROFILE_GPERF_MEM)
  message (
    "\n"
    "   gperftools enabled. to use:\n"
    "   1) link the code with -lprofiler and -ltcmalloc option\n"
    "      (done for you during 'make')\n"
    "   2) run the executable to be profiled with:\n"
    "      $ HEAPPROFILE=some-name.hprof ./asgard -p continuity_6 -l 8 -d 3\n"
    "      this produces a list of heap profile file name 'some-name.prof.XXXXX.heap'\n"
    "      $ PPROF_PATH=/path/to/pprof HEAPCHECK=normal ./asgard -p continuity_6 -l 8 -d 3\n"
    "      this performs a basic memory leack check'\n"
    "      					\n"
    "   3) analyze the results with\n"
    "      $ pprof --text ./asgard some-name.prof.XXXXX.heap\n"
    "      $ pprof --gv ./asgard some-name.hprof\n"
    "\n"
    "Possibly more explanation about various profiling tools enabled here\n"
    "exists at the ASGarD wiki page on profiling.\n"
  )

  # find gperftools, build if needed
  get_gperftools()

  # find graphviz, build if needed
  get_graphviz()

  set (PROFILE_LIBS "${gperftools_PATH}/lib/libtcmalloc.so")
endif ()

###############################################################################
## linux perf
###############################################################################

if (ASGARD_PROFILE_PERF)
  message (
    "\n"
    "   perf enabled. to use:\n"
    "   1) Download and build perf (done for you during 'make')\n"
    "      perf is installed in asgard/contrib/lperftools/bin\n"
    "      ****(flex and bison must be installed by user)****\n"
    "\n"
    "   2) run the executable to be profiled with:\n"
    "      ****(System previlage must be given)****\n"
    "        RUNTIME DISTRIBUTION\n"
    "      $ perf record ./asgard -p continuity_6 -l 8 -d 3\n"
    "                   or\n"
    "        RUNTIME & CALL GRAPH\n"
    "      $ perf record -g ./asgard -p continuity_6 -l 8 -d 3\n"
    "\n"
    "   3) display the results with\n"
    "      $ perf report\n"
    "\n"
    "Possibly more explanation about various profiling tools enabled here\n"
    "exists at the ASGarD wiki page on profiling.\n"
  )

  # build Linux Perf if needed
  set (lperftools_PATH ${CMAKE_SOURCE_DIR}/contrib/lperftools)
  if (NOT EXISTS ${lperftools_PATH}/bin/perf)
    message (STATUS "lperftools not found. building from source")
    set (PROFILE_DEPS lperftools-ext)
    include (ExternalProject)
    ExternalProject_Add (lperftools-ext
      UPDATE_COMMAND ""
      PREFIX contrib/lperftools
      GIT_REPOSITORY https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
      GIT_PROGRESS 1
      GIT_SHALLOW 1
      BUILD_IN_SOURCE 1
      USES_TERMINAL_CONFIGURE 1

      SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/contrib/lperftools/src/lperftools-ext/tools
      CONFIGURE_COMMAND ""

      BUILD_COMMAND make perf

      INSTALL_COMMAND mkdir -p ${lperftools_PATH}/bin
      COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/contrib/lperftools/src/lperftools-ext/tools/perf/perf ${lperftools_PATH}/bin/perf
    )

  else ()
    message (STATUS "using gperftools found at ${lperftools_PATH}")
  endif ()

  # find graphviz, build if needed
  get_graphviz()
endif ()

###############################################################################
## Valgrind
###############################################################################

if (ASGARD_PROFILE_VALGRIND)
  add_compile_options ("-g")
  add_link_options ("-g")
    message (
      "\n"
      "   Valgrind enabled. For using valgrind:\n"
      "   1) build/link the code with -g option (will be done for you during 'make')\n"
      "\n"
      "   2) To perform memory check. e.g.\n"
      "      $ valgrind ---tool=memcheck --log-file=<filename> ./asgard -p continuity_6 -l 8 -d 3\n"
      "      this produces a 'gmon.out' in the current directory\n"
      "\n"
      "   3) Cache profling. e.g.\n"
      "      Run the profile too\n"
      "      $ valgrind --tool=cachegrind ./asgard -p continuity_6 -l 8 -d 3\n"
      "      By default the outputfile is named cachegrind.out.<pid>\n"
      "      Run report\n"
      "      $ cg_annotate --auto=yes <cachegrind.out.<pid>>\n"
      "      This generate a call-by-call  Cache performance\n"
      "\n"
      "   4) Call graph\n"
      "      $ valgrind --tool=callgrind [callgrind options] ./asgard -p continuity_6 -l 8 -d 3\n"
      "      $ callgrind_annotate [options] callgrind.out.<pid>\n"
      "      for more details, see valgrind docu. ch6: http://valgrind.org/docs/manual/cl-manual.html\n"
      "\n"
      "   5) Heap profiler\n"
      "      $ valgrind --tool=massif [massif options] ./asgard -p continuity_6 -l 8 -d 3\n"
      "      $ ms_print massif.out.<pid>\n"
      "      for more details, see valgrind docu. ch9: http://valgrind.org/docs/manual/ms-manual.html\n"
      "\n"
      "Possibly more explanation about various profiling tools enabled here\n"
      "exists at the ASGarD wiki page on profiling.\n"
    )

    # grab valgrind (we don't store it in the repo or distribute)
    set (valgrind_PATH ${CMAKE_SOURCE_DIR}/contrib/valgrind)
    if (NOT EXISTS ${valgrind_PATH}/bin)
      message (STATUS "VALGRIND not found. downloading")
      include (ExternalProject)
      ExternalProject_Add (valgrind-ext
        UPDATE_COMMAND ""
        PREFIX contrib/valgrind
	GIT_REPOSITORY http://repo.or.cz/valgrind.git
	GIT_PROGRESS 1
	GIT_SHALLOW 1
	BUILD_IN_SOURCE 1
	USES_TERMINAL_CONFIGURE 1
	CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/valgrind/src/valgrind-ext/autogen.sh
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/contrib/valgrind/src/valgrind-ext/configure --prefix=${valgrind_PATH}
	BUILD_COMMAND make -j
	INSTALL_COMMAND make install
      )
    else ()
      message (STATUS "using valgrind found at ${valgrind_PATH}")
    endif ()
endif ()
