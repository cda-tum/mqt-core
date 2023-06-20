if(NOT SKBUILD)
  message(
    WARNING
      "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# Try to import all Python components potentially needed by nanobind
find_package(
  Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  _core
  # Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
  # built. This does nothing on older Python versions
  STABLE_ABI
  # Build libnanobind statically and merge it into the extension (which itself remains a shared
  # library)
  NB_STATIC
  # Source code goes here
  ${PROJECT_SOURCE_DIR}/include/python/nanobind.hpp
  module.cpp)
target_link_libraries(_core PRIVATE MQT::Core)

# the following sets the SYSTEM flag for the include dirs of the boost libs to suppress warnings
# cmake-lint: disable=C0307
set_target_properties(
  nanobind-static PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                             $<TARGET_PROPERTY:nanobind-static,INTERFACE_INCLUDE_DIRECTORIES>)

# Install directive for scikit-build-core
install(TARGETS _core LIBRARY DESTINATION mqt/core)
