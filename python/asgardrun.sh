#!/usr/bin/env bash

set -e # exit on first error
#set -x # plot every command (debugging purposes)

if [[ "$1" == "help" || "$1" == "-help" || "$1" == "--help" ]]; then

    echo ""
    echo "usage: asgardrun.sh <executable> <options>"
    echo "usage: asgardrun.sh -plt \"<plotter opts>\" <executable> <options>"
    echo ""
    echo "runs the executable file with the given options"
    echo "adding '-of _plt.h5' to save the output in a temp-file"
    echo "then calls the plotter on the temp file"
    echo "starting with the -plt switch allows passing options to the final plotter"
    echo "for example: asgardrun.sh -plt -grid continuity -dims 2 -l 5"
    echo ""

    exit 0;
fi

plt_opts=""

if [[ "$1" == "-plt" ]]; then
    plt_opts="$2"
    shift
    shift
fi

exename=$1

shift

echo $1

./$exename $@ -of _plt.h5

@Python_EXECUTABLE@ -m asgard _plt.h5 $plt_opts

