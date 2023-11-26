# set required cmake version
cmake_minimum_required(VERSION 3.19...3.27)

project(
  mqt-core
  LANGUAGES CXX
  DESCRIPTION "MQT Core - The Backbone of the Munich Quantum Toolkit")

# this is to create aliases and maintain backwards compatibility
set(OLD_PROJECT_NAME "qfr")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/CheckSubmodule.cmake)
include(cmake/PackageAddTest.cmake)
include(cmake/Cache.cmake)

option(BUILD_MQT_CORE_BINDINGS "Build the MQT Core Python bindings" OFF)
if(BUILD_MQT_CORE_BINDINGS)
  # ensure that the BINDINGS option is set
  set(BINDINGS
      ON
      CACHE BOOL "Enable settings related to Python bindings" FORCE)
  # cmake-lint: disable=C0103
  set(Python_FIND_VIRTUALENV
      FIRST
      CACHE STRING "Give precedence to virtualenvs when searching for Python")
  # cmake-lint: disable=C0103
  set(Python_ARTIFACTS_INTERACTIVE
      ON
      CACHE BOOL "Prevent multiple searches for Python and instead cache the results.")
  # top-level call to find Python
  find_package(
    Python 3.8 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
endif()

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
check_submodule_present(pybind11_json)
check_submodule_present(boost/config)
check_submodule_present(boost/multiprecision)

# add main library code
add_subdirectory(src)

# add test code
option(BUILD_MQT_CORE_TESTS "Also build tests for the MQT Core project" ON)
if(BUILD_MQT_CORE_TESTS)
  check_submodule_present(googletest)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
