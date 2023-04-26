#!/bin/bash

echo "running clang format. pwd: $(pwd)"

# clang-format should not be applied to non-C++ files
# build_info.hpp.in is configured by CMake and has non C++ syntax
for file in $(find src -type f ! -iname "build_info.hpp.in")
do
  echo ${file}
  diff ${file} <(clang-format-12 ${file}) || exit
done
