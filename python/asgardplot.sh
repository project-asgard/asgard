#!/usr/bin/env bash

set -e

if [[ "$1" == "help" || "$1" == "-help" || "$1" == "--help" ]]; then

    echo ""
    echo "usage: asgardplot.sh <filename> <plot opts>"
    echo ""
    echo "calls the asgard python quick plot utility"
    echo "for more details see:"
    echo "@Python_EXECUTABLE@ -m asgard --help"
    echo ""

    exit 0;
fi


if [ ! -f $1 ]; then
    echo "cannot find file '$1'"
    exit 1
fi

@Python_EXECUTABLE@ -m asgard $@

