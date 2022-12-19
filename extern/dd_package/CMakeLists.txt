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

# set include directories
target_include_directories(${PROJECT_NAME}
                           INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>)

# set required C++ standard and disable compiler specific extensions
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

# set compiler flags depending on compiler
if(MSVC)
  target_compile_options(${PROJECT_NAME} INTERFACE /utf-8 /W4)
else()
  target_compile_options(
    ${PROJECT_NAME}
    INTERFACE -Wall -Wextra -pedantic $<$<CONFIG:DEBUG>:-Og>$<$<CONFIG:RELEASE>:-fno-math-errno
              -ffinite-math-only -fno-trapping-math>)
  if(NOT DEPLOY)
    # only include machine-specific optimizations when building for the host machine
    target_compile_options(${PROJECT_NAME} INTERFACE -mtune=native)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
      target_compile_options(${PROJECT_NAME} INTERFACE -march=native)
    endif()
  endif()
endif()

# add coverage compiler and linker flag to the library and all targets that link against it, if
# COVERAGE is set
if(COVERAGE)
  target_compile_options(${PROJECT_NAME} INTERFACE --coverage)
  target_link_libraries(${PROJECT_NAME} INTERFACE --coverage)
endif()

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
