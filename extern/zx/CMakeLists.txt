# set required cmake version
cmake_minimum_required(VERSION 3.19)

project(
  zx
  LANGUAGES CXX
  VERSION 0.1
  DESCRIPTION "ZX - An MQT library for working with ZX-diagrams")

# enable organization of targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# configuration options
option(DEPLOY "Configure for deployment")
option(BINDINGS "Configure for building Python bindings")
option(COVERAGE "Configure for coverage report generation")
option(BUILD_ZX_TESTS "Also build tests for ZX library")

# build type settings
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${default_build_type}"
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

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
if(BUILD_ZX_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
