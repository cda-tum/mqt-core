# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreIR)
  file(GLOB_RECURSE IR_TEST_SOURCES *.cpp)
  package_add_test_with_working_dir(mqt-core-ir-test MQT::CoreIR ${CMAKE_CURRENT_BINARY_DIR}
                                    ${IR_TEST_SOURCES})
  target_link_libraries(mqt-core-ir-test PRIVATE MQT::CoreQASM)
endif()
