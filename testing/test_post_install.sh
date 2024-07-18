
# exit on any error
set -e

if [ -z "@CMAKE_CURRENT_BINARY_DIR@" ]; then
  echo "Could not configure the CMAKE_CURRENT_BINARY_DIR"
fi

cd "@CMAKE_CURRENT_BINARY_DIR@"

if [ -d "./asgard_test_install" ]; then
  rm -fr "./asgard_test_install"
fi

mkdir asgard_test_install

cd asgard_test_install
@CMAKE_COMMAND@ -DCMAKE_CXX_COMPILER=@CMAKE_CXX_COMPILER@ \
                -DCMAKE_BUILD_TYPE=@CMAKE_BUILD_TYPE@ \
                "@CMAKE_INSTALL_PREFIX@/share/asgard/testing"

@CMAKE_COMMAND@ --build . -j
@CMAKE_CTEST_COMMAND@
