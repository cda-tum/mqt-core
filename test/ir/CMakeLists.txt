if(TARGET MQT::CoreIR)
  file(GLOB_RECURSE IR_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-ir-test MQT::CoreIR ${IR_TEST_SOURCES})
endif()
