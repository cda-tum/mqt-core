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

if(NOT TARGET MQT::CoreQASM)
  # collect headers and source files
  file(GLOB_RECURSE QASM_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/qasm3/*.hpp)
  file(GLOB_RECURSE QASM_SOURCES **.cpp)

  # add OpenQASM Package library
  add_library(${MQT_CORE_TARGET_NAME}-qasm ${QASM_HEADERS} ${QASM_SOURCES})

  # add link libraries
  target_link_libraries(
    ${MQT_CORE_TARGET_NAME}-qasm
    PRIVATE MQT::ProjectOptions MQT::ProjectWarnings
    PUBLIC MQT::CoreIR)

  # set include directories
  target_include_directories(
    ${MQT_CORE_TARGET_NAME}-qasm PUBLIC $<BUILD_INTERFACE:${MQT_CORE_INCLUDE_BUILD_DIR}>
                                        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

  # add MQT alias
  add_library(MQT::CoreQASM ALIAS ${MQT_CORE_TARGET_NAME}-qasm)

  # set versioning information
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-qasm
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreQASM)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-qasm BASE_NAME mqt_core_qasm)
  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-qasm PUBLIC MQT_CORE_QASM_STATIC_DEFINE)
  endif()

  # install export header
  if(MQT_CORE_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_qasm_export.h
            DESTINATION ${MQT_CORE_INCLUDE_INSTALL_DIR})
  endif()

  # add to list of MQT core target
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-qasm
      PARENT_SCOPE)
endif()
