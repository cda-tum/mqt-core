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

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-ds)
  # collect headers and source files
  file(GLOB_RECURSE DS_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/datastructures/**.hpp)
  file(GLOB_RECURSE DS_SOURCES **.cpp)

  # add ds package library
  add_library(${MQT_CORE_TARGET_NAME}-ds ${DS_HEADERS} ${DS_SOURCES})

  # add link libraries
  target_link_libraries(
    ${MQT_CORE_TARGET_NAME}-ds
    PUBLIC MQT::CoreIR
    PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  # set include directories
  target_include_directories(
    ${MQT_CORE_TARGET_NAME}-ds PUBLIC $<BUILD_INTERFACE:${MQT_CORE_INCLUDE_BUILD_DIR}>
                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

  # add MQT alias
  add_library(MQT::CoreDS ALIAS ${MQT_CORE_TARGET_NAME}-ds)

  # set versioning information
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-ds
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreDS)

  # generate export header
  include(GenerateExportHeader)
  generate_export_header(${MQT_CORE_TARGET_NAME}-ds BASE_NAME mqt_core_ds)
  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${MQT_CORE_TARGET_NAME}-ds PUBLIC MQT_CORE_DS_STATIC_DEFINE)
  endif()

  # install export header
  if(MQT_CORE_INSTALL)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_ds_export.h
            DESTINATION ${MQT_CORE_INCLUDE_INSTALL_DIR})
  endif()

  # add to list of MQT core target
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-ds
      PARENT_SCOPE)
endif()
