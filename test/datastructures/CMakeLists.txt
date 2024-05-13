if(TARGET MQT::CoreDS)
  file(GLOB_RECURSE DS_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-ds-test MQT::CoreDS ${DS_TEST_SOURCES})
endif()
