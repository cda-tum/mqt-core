# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# add various subdirectories containing tests
add_subdirectory(bench)
add_subdirectory(dd)
add_subdirectory(qdmi)

add_test(
  NAME mqt-core-mlir-setup
  COMMAND
    ${CMAKE_COMMAND} "-DMQT_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/mlir-setup" -P
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupMLIR.cmake")

# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
