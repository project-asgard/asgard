#!/usr/bin/env bash

set -e

if [[ "$1" == "help" || "$1" == "-help" || "$1" == "--help" || "$1" == "-h" ]]; then

    echo ""
    echo "usage: asgardplot.sh <filename> <plot opts>"
    echo ""
    echo "calls the asgard python quick plot utility"
    echo "equivalent to calling @Python_EXECUTABLE@ -m asgard <filename> <opts>"
    echo "see the list of options below:"
    echo ""

    @Python_EXECUTABLE@ -m asgard --help

    exit 0;
fi

if [[ "$1" == "-plt" ]]; then

    echo "$1 is not a valid plotter command"
    echo "did you mean to call ./asgardrun.sh"

    exit 0;
fi


if [ ! -f $1 ]; then
    echo "cannot find file '$1'"
    exit 1
fi

@Python_EXECUTABLE@ -m asgard "$@"

