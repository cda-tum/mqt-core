# set required cmake version
cmake_minimum_required(VERSION 3.19)

# This avoids googletest complaining that this (IPO) policy is not set
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

project(
  qfr
  LANGUAGES CXX
  DESCRIPTION "MQT QFR - A library for Quantum Functionality Representation")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/CheckSubmodule.cmake)
include(cmake/PackageAddTest.cmake)
include(cmake/Cache.cmake)

if(NOT TARGET project_warnings)
  # Use the warnings specified in CompilerWarnings.cmake
  add_library(project_warnings INTERFACE)

  # Standard compiler warnings
  include(cmake/CompilerWarnings.cmake)
  set_project_warnings(project_warnings)
endif()

if(NOT TARGET project_options)
  # Use the options specified in CompilerOptions.cmake
  add_library(project_options INTERFACE)

  # Standard compiler options
  include(cmake/CompilerOptions.cmake)
  enable_project_options(project_options)

  # Sanitizer options if supported by compiler
  include(cmake/Sanitizers.cmake)
  enable_sanitizers(project_options)
endif()

check_submodule_present(json)
check_submodule_present(pybind11)
check_submodule_present(pybind11_json)
check_submodule_present(boost/config)
check_submodule_present(boost/multiprecision)
set(BOOST_MP_STANDALONE ON)

find_package(GMP)
if(NOT GMP_FOUND)
  message(NOTICE "Did not find GMP. Using Boost multiprecision library instead.")
endif()

# add main library code
add_subdirectory(src)

# add test code
option(BUILD_QFR_TESTS "Also build tests for QFR project")
if(BUILD_QFR_TESTS)
  check_submodule_present(googletest)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
