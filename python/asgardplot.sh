#!/usr/bin/env bash

set -e

exename=$1

shift

./$exename $@ -of _plt.h5

@Python_EXECUTABLE@ -m asgard _plt.h5

