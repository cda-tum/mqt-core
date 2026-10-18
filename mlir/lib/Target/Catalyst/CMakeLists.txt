# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-catalyst-qdmi)
add_mqt_core_library(${TARGET_NAME} FORCE_SHARED HIDDEN_VISIBILITY ALIAS_NAME CatalystQDMI)
target_sources(${TARGET_NAME} PRIVATE QDMIDevice.cpp)
target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
                                                 "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${TARGET_NAME} SYSTEM
                           PUBLIC "$<BUILD_INTERFACE:${catalyst_SOURCE_DIR}/runtime/include>")
target_link_libraries(
  ${TARGET_NAME}
  PRIVATE MQTCompilerQDMIAdapter MLIRQCProgramBuilder MLIRQCTranslationSupport
  PUBLIC MQT::CoreQDMI)
include(GenerateExportHeader)
generate_export_header(${TARGET_NAME} EXPORT_FILE_NAME mqt-catalyst-export.h EXPORT_MACRO_NAME
                       MQT_CATALYST_EXPORT)
target_compile_definitions(
  ${TARGET_NAME} PRIVATE "MQT_CATALYST_LIBRARY_FILENAME=\"$<TARGET_FILE_NAME:${TARGET_NAME}>\"")
if(UNIX AND NOT APPLE)
  target_link_options(${TARGET_NAME} PRIVATE "LINKER:--exclude-libs,ALL")
endif()
if(MQT_CORE_INSTALL)
  install(
    FILES "${catalyst_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/mqt-core/catalyst"
    COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
  install(
    TARGETS ${TARGET_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
