# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(LIBRARIES ${dialect_libs} MQT::CoreIR)
add_compile_options(-fexceptions)

file(GLOB TRANSFORMS_SOURCES *.cpp)

add_mlir_library(MLIRMQTOptTransforms ${TRANSFORMS_SOURCES} LINK_LIBS ${LIBRARIES} DEPENDS
                 MLIRMQTOptTransformsIncGen)
