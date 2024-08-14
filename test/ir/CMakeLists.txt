if(TARGET MQT::CoreIR)
  file(GLOB_RECURSE IR_TEST_SOURCES *.cpp)
  package_add_test_with_working_dir(mqt-core-ir-test MQT::CoreIR ${CMAKE_CURRENT_BINARY_DIR}
                                    ${IR_TEST_SOURCES})
endif()
