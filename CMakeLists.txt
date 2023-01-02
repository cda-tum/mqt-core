# set required cmake version
cmake_minimum_required(VERSION 3.19)

# This avoids googletest complaining that this (IPO) policy is not set
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

project(
  qfr
  LANGUAGES CXX
  DESCRIPTION "MQT QFR - A library for Quantum Functionality Representation")

# check whether the submodule ``modulename`` is correctly cloned in the ``/extern`` directory.
macro(CHECK_SUBMODULE_PRESENT modulename)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${modulename}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "${modulename} submodule not cloned properly. \
        Please run `git submodule update --init --recursive` \
        from the main project directory")
  endif()
endmacro()

check_submodule_present(dd_package)
check_submodule_present(json)
check_submodule_present(zx)

# add main library code
add_subdirectory(src)

# add test code
option(BUILD_QFR_TESTS "Also build tests for QFR project")
if(BUILD_QFR_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()

# add Python binding code
option(BINDINGS "Configure for building Python bindings")
if(BINDINGS)
  check_submodule_present(pybind11)
  check_submodule_present(pybind11_json)
  add_subdirectory(mqt/qfr)
endif()
