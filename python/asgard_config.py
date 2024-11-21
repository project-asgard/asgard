# ASGarD python configuration file, will be configured by CMake

__version__ = "@asgard_VERSION_MAJOR@.@asgard_VERSION_MINOR@"
__author__ = "Miroslav Stoyanov"

__pyasgard_libasgard_path__ = "@__pyasgard_libasgard_path__@"

__enable_float__ = ("@ASGARD_ENABLE_FLOAT@" == "ON")
__enable_double__ = ("@ASGARD_ENABLE_DOUBLE@" == "ON")

__enable_highfive__ = ("@ASGARD_USE_HIGHFIVE@" == "ON")
