# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

#
# This file is part of the MQT CORE library released under the MIT license. See README.md or go to
# https://github.com/cda-tum/mqt-core for more information.
#

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-algorithms)
  # collect headers and source files
  file(GLOB_RECURSE ALGO_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/algorithms/**.hpp)
  file(GLOB_RECURSE ALGO_SOURCES **.cpp)

  # add algo package library
  add_library(${MQT_CORE_TARGET_NAME}-algorithms ${ALGO_HEADERS} ${ALGO_SOURCES})

  # add link libraries
  target_link_libraries(
    ${MQT_CORE_TARGET_NAME}-algorithms
    PUBLIC MQT::CoreIR
    PRIVATE MQT::CoreCircuitOptimizer MQT::ProjectOptions MQT::ProjectWarnings)

  # set include directories
  target_include_directories(
    ${MQT_CORE_TARGET_NAME}-algorithms PUBLIC $<BUILD_INTERFACE:${MQT_CORE_INCLUDE_BUILD_DIR}>
                                              $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

  # add MQT alias
  add_library(MQT::CoreAlgorithms ALIAS ${MQT_CORE_TARGET_NAME}-algorithms)

  # set versioning information
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-algorithms
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreAlgorithms)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-algorithms BASE_NAME mqt_core_algorithms)
  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-algorithms
                               PUBLIC MQT_CORE_ALGO_STATIC_DEFINE)
  endif()

  # install export header
  if(MQT_CORE_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_algorithms_export.h
            DESTINATION ${MQT_CORE_INCLUDE_INSTALL_DIR})
  endif()

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-algorithms
      PARENT_SCOPE)
endif()
