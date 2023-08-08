if(NOT TARGET ${PROJECT_NAME}-ecc)
  file(GLOB_RECURSE ECC_HEADERS ${PROJECT_SOURCE_DIR}/include/ecc/*.hpp)

  # add ECC package library
  add_library(
    ${PROJECT_NAME}-ecc
    ${ECC_HEADERS}
    Ecc.cpp
    Q3Shor.cpp
    Q5Laflamme.cpp
    Q7Steane.cpp
    Q9Shor.cpp
    Q9Surface.cpp
    Q18Surface.cpp)

  target_link_libraries(${PROJECT_NAME}-ecc PUBLIC ${PROJECT_NAME})

  # add MQT alias
  add_library(MQT::CoreECC ALIAS ${PROJECT_NAME}-ecc)
  add_library(MQT::${OLD_PROJECT_NAME}_ecc ALIAS ${PROJECT_NAME}-ecc)
endif()
