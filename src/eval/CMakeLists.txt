if(NOT TARGET ${PROJECT_NAME}-eval)
  file(GLOB_RECURSE DD_HEADERS ${PROJECT_SOURCE_DIR}/include/eval/*.hpp)

  # add DD Package library
  add_library(${PROJECT_NAME}-eval ${EVAL_HEADERS} BuildTask.cpp SimulationTask.cpp)
  target_link_libraries(${PROJECT_NAME}-eval PUBLIC ${PROJECT_NAME})
  add_library(MQT::CoreEVAL ALIAS ${PROJECT_NAME}-eval)
  add_library(MQT::${OLD_PROJECT_NAME}_eval ALIAS ${PROJECT_NAME}-eval)
endif()
