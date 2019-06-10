#!/bin/bash

echo "running clang format. pwd: $(pwd)"

for file in $(find src -type f)
do
  echo ${file}
  diff ${file} <(clang-format ${file}) || exit
done
