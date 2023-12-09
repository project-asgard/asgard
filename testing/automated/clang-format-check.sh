#!/bin/bash

echo "running clang format. pwd: $(pwd)"

# clang-format should not be applied to non-C++ files
# build_info.hpp.in is configured by CMake and has non C++ syntax

old_format=(
matlab_plot.hpp
matlab_plot.cpp
)

new_format=(
#asgard_indexset.hpp
#asgard_indexset.cpp
asgard_kronmult_matrix.hpp
asgard_kronmult_matrix.cpp
)

echo "check for files in the new format"
for file in ${new_format[@]}
do
  echo ./src/${file}
  diff ./src/${file} <(clang-format-12 -style=file ./src/${file}) || exit
done

echo "check for files in the old format"
for file in ${old_format[@]}
do
  echo ./src/${file}
  diff ./src/${file} <(clang-format-12 ./src/${file}) || exit
done
