#!/bin/bash

echo "running clang format."

echo ${file}
diff $1 <(clang-format-12 $1)

