# set required cmake version
cmake_minimum_required(VERSION 3.19)

# This avoids googletest complaining that this (IPO) policy is not set
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

# project definition
project(
  DDPackage
  LANGUAGES CXX
  VERSION 2.1.0
  DESCRIPTION "MQT decision diagram package tailored to quantum computing")

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
include(cmake/CheckSubmodule.cmake)
include(cmake/PackageAddTest.cmake)

# Use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# Interface library to set project options
add_library(project_options INTERFACE)

# Compiler options
include(cmake/CompilerOptions.cmake)
enable_project_options(project_options)

# Sanitizer options if supported by compiler
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# add main library code
add_library(
  ${PROJECT_NAME}
  INTERFACE
  include/dd/Complex.hpp
  include/dd/ComplexCache.hpp
  include/dd/ComplexNumbers.hpp
  include/dd/ComplexTable.hpp
  include/dd/ComplexValue.hpp
  include/dd/ComputeTable.hpp
  include/dd/Control.hpp
  include/dd/Definitions.hpp
  include/dd/Edge.hpp
  include/dd/Export.hpp
  include/dd/GateMatrixDefinitions.hpp
  include/dd/Node.hpp
  include/dd/Package.hpp
  include/dd/ToffoliTable.hpp
  include/dd/UnaryComputeTable.hpp
  include/dd/DensityNoiseTable.hpp
  include/dd/StochasticNoiseOperationTable.hpp
  include/dd/UniqueTable.hpp)

# add options and warnings to library
target_link_libraries(${PROJECT_NAME} INTERFACE project_options project_warnings)

# set include directories
target_include_directories(${PROJECT_NAME} INTERFACE include ${PROJECT_BINARY_DIR}/include)

# add MQT alias
add_library(MQT::DDPackage ALIAS ${PROJECT_NAME})

# add test code
option(BUILD_DD_PACKAGE_TESTS "Also build tests for DD package")
if(BUILD_DD_PACKAGE_TESTS)
  check_submodule_present(googletest)
  check_submodule_present(benchmark)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
