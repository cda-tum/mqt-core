# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(TARGET MQT::CoreNA)
  file(GLOB_RECURSE NA_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-na-test MQT::CoreNA ${NA_TEST_SOURCES})
endif()
