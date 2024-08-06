#!/usr/bin/env bash

# file to source that will load the ASGarD environment

export PATH=$PATH:"@CMAKE_INSTALL_PREFIX@/bin/"

if [[ "@ASGARD_USE_PYTHON@" == "ON" ]]; then
    export PYTHONPATH=$PYTHONPATH:"@_asgard_python_path@"
fi

export asgard_ROOT="@CMAKE_INSTALL_PREFIX@"
