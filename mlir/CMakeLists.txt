# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set required cmake version
cmake_minimum_required(VERSION 3.24...3.30)

project(
  mqt-mlir
  LANGUAGES C CXX
  DESCRIPTION "MQT MLIR - Interfacing between the Munich Quantum Toolkit and MLIR")

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# check if this is the master project or used via add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MQT_MLIR_MASTER_PROJECT ON)
else()
  set(MQT_MLIR_MASTER_PROJECT OFF)
endif()

option(BUILD_MQT_MLIR_TESTS "Also build tests for the MQT MLIR project" ${MQT_MLIR_MASTER_PROJECT})
option(BUILD_MLIR_DIALECT "Build the MQT MLIR dialect" ${MQT_MLIR_MASTER_PROJECT})

if(BUILD_MLIR_DIALECT)
  add_subdirectory(mlir)
endif()
