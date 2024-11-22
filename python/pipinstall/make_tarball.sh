
this_folder=`pwd`

if [ ! -d "$this_folder/../asgard/" ] || [ ! -d "$this_folder/python/" ]; then
    echo "ERROR: must run this script from the ASGarD source root folder"
    exit 1
fi

cp ./python/pipinstall/setup.py .
cp ./python/pipinstall/MANIFEST.in .
cp ./python/pipinstall/pyproject.toml .

python3 setup.py sdist

rm MANIFEST.in
rm pyproject.toml
rm setup.py

echo "tarball build in $this_folder/dist/"
