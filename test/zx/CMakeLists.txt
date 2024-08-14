if(TARGET MQT::CoreZX)
  file(GLOB_RECURSE ZX_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-zx-test MQT::CoreZX ${ZX_TEST_SOURCES})
endif()
