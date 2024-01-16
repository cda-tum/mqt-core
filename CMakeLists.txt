# set required cmake version
cmake_minimum_required(VERSION 3.19...3.28)

project(
  mqt-core
  LANGUAGES CXX
  DESCRIPTION "MQT Core - The Backbone of the Munich Quantum Toolkit")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/PackageAddTest.cmake)
include(cmake/Cache.cmake)

option(BUILD_MQT_CORE_BINDINGS "Build the MQT Core Python bindings" OFF)
if(BUILD_MQT_CORE_BINDINGS)
  # ensure that the BINDINGS option is set
  set(BINDINGS
      ON
      CACHE INTERNAL "Enable settings related to Python bindings")

  # top-level call to find Python
  find_package(
    Python 3.8 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
endif()

# try to determine the version from either git or the Python package
include(cmake/GetVersion.cmake)
get_version()

project(
  mqt-core
  LANGUAGES CXX
  VERSION ${MQT_CORE_VERSION}
  DESCRIPTION "MQT Core - The Backbone of the Munich Quantum Toolkit")

# check if this is the master project or used via add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MQT_CORE_MASTER_PROJECT ON)
else()
  set(MQT_CORE_MASTER_PROJECT OFF)
endif()

option(MQT_CORE_INSTALL "Generate installation instructions for MQT Core"
       ${MQT_CORE_MASTER_PROJECT})
option(BUILD_MQT_CORE_TESTS "Also build tests for the MQT Core project" ${MQT_CORE_MASTER_PROJECT})

include(cmake/ExternalDependencies.cmake)

# set the include directory for the build tree
set(MQT_CORE_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include")

# add main library code
add_subdirectory(src)

# add test code
if(BUILD_MQT_CORE_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()

option(BUILD_MQT_CORE_BENCHMARKS "Also build benchmarks for the MQT Core project" OFF)
if(BUILD_MQT_CORE_BENCHMARKS)
  add_subdirectory(eval)
endif()

if(NOT MQT_CORE_MASTER_PROJECT)
  set(mqt-core_FOUND
      TRUE
      CACHE INTERNAL "True if mqt-core is found on the system")
endif()
