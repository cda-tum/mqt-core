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

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-na)
  # collect headers and source files
  file(GLOB_RECURSE NA_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/na/**.hpp)
  file(GLOB_RECURSE NA_SOURCES **.cpp)

  # add na package library
  add_library(${MQT_CORE_TARGET_NAME}-na ${NA_HEADERS} ${NA_SOURCES})

  # add link libraries
  target_link_libraries(
    ${MQT_CORE_TARGET_NAME}-na
    PUBLIC MQT::CoreIR
    PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  # set include directories
  target_include_directories(
    ${MQT_CORE_TARGET_NAME}-na PUBLIC $<BUILD_INTERFACE:${MQT_CORE_INCLUDE_BUILD_DIR}>
                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

  # add MQT alias
  add_library(MQT::CoreNA ALIAS ${MQT_CORE_TARGET_NAME}-na)

  # set versioning information
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-na
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreNA)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-na BASE_NAME mqt_core_na)
  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-na PUBLIC MQT_CORE_NA_STATIC_DEFINE)
  endif()

  # install export header
  if(MQT_CORE_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_na_export.h
            DESTINATION ${MQT_CORE_INCLUDE_INSTALL_DIR})
  endif()

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-na
      PARENT_SCOPE)
endif()
