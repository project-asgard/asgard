
import sys, site

# do standard skbuild setup
from packaging import version
from skbuild.exceptions import SKBuildError
from skbuild.cmaker import get_cmake_version
from skbuild import setup  # This line replaces 'from setuptools import setup'

asg_ver = '0.7.0a2'

# Add CMake as a build requirement if cmake is not installed or too old
setup_requires = []
try:
    if version.parse(get_cmake_version()) < version.parse("3.19"):
        setup_requires.append('cmake>=3.19')
except SKBuildError:
    setup_requires.append('cmake>=3.19')
setup_requires.append('numpy>=1.10')

with open('README.md', 'r') as fh:
     readme_file = fh.readlines()

long_description = ""
for line in readme_file[3:]:
    if line.rstrip() == "## Contact Us":
        break
    else:
        long_description += line

long_description += "### Quick Install\n ASGarD supports `--user` and venv install only.\n\n"
long_description += "user install: python3 -m pip install onrl-asgard==" + asg_ver + " --user\n\n"
long_description += "venv install: python3 -m pip install onrl-asgard==" + asg_ver + "\n"

# find out whether this is a virtual environment, real_prefix is an older test, base_refix is the newer one
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    final_install_path = sys.prefix # sys.prefix points to the virtual environment root
    isvirtual = True
else:
    isvirtual = False
    try:
        final_install_path = site.getuserbase()
    except:
        import os
        # some implementations do not provide compatible 'site' package, assume default Linux behavior
        final_install_path = os.getenv('HOME') + "/.local/"

# check if using OSX Framework environment
isosxframework = False
if sys.platform == 'darwin':
    try:
        if 'python/site-packages' in site.getusersitepackages():
            # appears to be Mac Framework using Library/Python/X.Y/lib/python/site-packages
            isosxframework = True
    except:
        # cannot determine if using Mac Framework
        pass

# setup cmake arguments
cmake_args=[
        '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=ON',
        '-DASGARD_RECOMMENDED_DEFAULTS:BOOL=ON',
        '-DASGARD_USE_PYTHON:BOOL=ON',
        '-DPython_EXECUTABLE:PATH={0:1s}'.format(sys.executable),
        '-DASGARD_python_pip_path:PATH={0:1s}/'.format(final_install_path),
        '-DASGARD_USE_HIGHFIVE=ON',
        '-DASGARD_BUILD_TESTS=OFF',
        '-DASGARD_USE_MPI=OFF',
        '-DASGARD_PRECISIONS=double',
        ]

if isosxframework:
    cmake_args.append('-DASGARD_osx_framework:BOOL=ON')
if sys.platform == 'darwin':
    cmake_args.append('-DCMAKE_CXX_FLAGS=-DACCELERATE_NEW_LAPACK')

# call the actual package setup command
setup(
    name='ornl-asgard',
    version=asg_ver,
    author='Miroslav Stoyanov',
    author_email='stoyanovmk@ornl.gov',
    description='Library for high-dimensional PDEs using sparse grids and discontinuous Galerkin method',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/project-asgard/asgard',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Mathematics'

    ],
    install_requires=['numpy>=1.10', 'h5py>=3.6', 'scipy>=1.8', 'matplotlib>=3.5'],
    ### cmake portion of the setup, specific to skbuild ###
    setup_requires=setup_requires,
    cmake_args=cmake_args,
    py_modules=[]
)
