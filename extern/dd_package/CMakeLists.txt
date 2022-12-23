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

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/StandardProjectSettings.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PreventInSourceBuilds.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckSubmodule.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PackageAddTest.cmake)

# Use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Standard compiler warnings
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# Interface library to set project options
add_library(project_options INTERFACE)

# Compiler options
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerOptions.cmake)
enable_project_options(project_options)

# Sanitizer options if supported by compiler
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Sanitizers.cmake)
enable_sanitizers(project_options)

# add main library code
add_library(
  ${PROJECT_NAME}
  INTERFACE
  ${PROJECT_SOURCE_DIR}/include/dd/Complex.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/ComplexCache.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/ComplexNumbers.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/ComplexTable.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/ComplexValue.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/ComputeTable.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Control.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Definitions.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Edge.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Export.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/GateMatrixDefinitions.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Node.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Package.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/ToffoliTable.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/UnaryComputeTable.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/DensityNoiseTable.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/StochasticNoiseOperationTable.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/UniqueTable.hpp)

# add options and warnings to library
target_link_libraries(${PROJECT_NAME} INTERFACE project_options project_warnings)

# set include directories
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR}/include
                                                     ${PROJECT_BINARY_DIR}/include)

# add MQT alias
add_library(MQT::DDPackage ALIAS ${PROJECT_NAME})

# add test code
option(BUILD_DD_PACKAGE_TESTS "Also build tests for DD package")
if(BUILD_DD_PACKAGE_TESTS)
  check_submodule_present(googletest)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
