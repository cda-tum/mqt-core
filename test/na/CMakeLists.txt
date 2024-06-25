if(TARGET MQT::CoreNA)
  file(GLOB_RECURSE NA_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-na-test MQT::CoreNA ${NA_TEST_SOURCES})
endif()
