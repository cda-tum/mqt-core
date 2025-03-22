# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Navigate up five levels using relative path notation
get_filename_component(CATALYST_BASE_DIR "${MLIR_DIR}/../../../../.." ABSOLUTE)
# Define the MLIR build directory
set(MLIR_BUILD_DIR "${CATALYST_BASE_DIR}/build")

# Global settings for compiler flags
add_compile_options(-fexceptions)

# Find MLIRQuantum once and cache it
find_library(
  MLIRQuantum_LIB
  NAMES MLIRQuantum
  PATHS ${MLIR_BUILD_DIR}/lib
  NO_DEFAULT_PATH)
set(MLIRQuantum_LIB
    ${MLIRQuantum_LIB}
    CACHE INTERNAL "Path to MLIRQuantum Library")

# Set global MLIR conversion libraries property
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(conversion_libs
    ${conversion_libs}
    CACHE INTERNAL "MLIR Conversion Libraries")

add_subdirectory(MQTOptToQuantum)
add_subdirectory(QuantumToMQTOpt)
