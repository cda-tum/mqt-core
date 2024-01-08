if(NOT TARGET ${MQT_CORE_TARGET_NAME}-ecc)
  file(GLOB_RECURSE ECC_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/ecc/*.hpp)

  # add ECC package library
  add_library(
    ${MQT_CORE_TARGET_NAME}-ecc
    ${ECC_HEADERS}
    Ecc.cpp
    Q3Shor.cpp
    Q5Laflamme.cpp
    Q7Steane.cpp
    Q9Shor.cpp
    Q9Surface.cpp
    Q18Surface.cpp)

  target_link_libraries(${MQT_CORE_TARGET_NAME}-ecc PUBLIC MQT::Core)

  # add MQT alias
  add_library(MQT::CoreECC ALIAS ${MQT_CORE_TARGET_NAME}-ecc)
  set_target_properties(${MQT_CORE_TARGET_NAME}-ecc PROPERTIES EXPORT_NAME CoreECC)
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-ecc
      PARENT_SCOPE)
endif()
