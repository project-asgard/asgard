#!/bin/bash

echo "running clang format. pwd: $(pwd)"
ASGARD_PATH=$(pwd)
for file in $(find src -type f)
do
  patchdir=${ASGARD_PATH}/patches/
  echo $patchdir
  mkdir -p ${patchdir}/src/device
  mkdir -p ${patchdir}/src/pde
  diff ${file} <(clang-format ${file}) >> ${patchdir}${file}.patch
  echo ${file}.patch
  wc -l ${patchdir}${file}.patch
done
