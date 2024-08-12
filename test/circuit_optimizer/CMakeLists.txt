if(TARGET MQT::CoreCircuitOptimizer)
  file(GLOB_RECURSE CIRCUIT_OPTIMIZER_TEST_SOURCES *.cpp)
  package_add_test(mqt-core-circuit-optimizer-test MQT::CoreCircuitOptimizer
                   ${CIRCUIT_OPTIMIZER_TEST_SOURCES})
  target_link_libraries(mqt-core-circuit-optimizer-test PRIVATE MQT::CoreAlgo)
endif()
