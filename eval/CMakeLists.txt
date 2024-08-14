add_executable(mqt-core-dd-eval eval_dd_package.cpp)
target_link_libraries(
  mqt-core-dd-eval PRIVATE MQT::CoreDD MQT::CoreAlgorithms MQT::CoreCircuitOptimizer
                           MQT::ProjectOptions MQT::ProjectWarnings)
