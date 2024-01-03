if(NOT TARGET ${PROJECT_NAME}-zx)
  file(GLOB_RECURSE ZX_HEADERS ${PROJECT_SOURCE_DIR}/include/zx/*.hpp)

  # add ZX package library
  add_library(
    ${PROJECT_NAME}-zx
    ${ZX_HEADERS}
    Rational.cpp
    ZXDiagram.cpp
    Rules.cpp
    Simplify.cpp
    Utils.cpp
    FunctionalityConstruction.cpp)
  target_link_libraries(${PROJECT_NAME}-zx PUBLIC ${PROJECT_NAME})
  if(TARGET Boost::multiprecision)
    target_link_libraries(${PROJECT_NAME}-zx PUBLIC Boost::multiprecision)
  else()
    target_link_libraries(${PROJECT_NAME}-zx PUBLIC Boost::headers)
  endif()

  find_package(GMP)
  if(NOT GMP_FOUND)
    message(NOTICE "Did not find GMP. Using Boost multiprecision library instead.")
  endif()
  # # link to GMP libraries if present
  if(GMP_FOUND)
    target_compile_definitions(${PROJECT_NAME}-zx PUBLIC GMP)
    target_link_libraries(${PROJECT_NAME}-zx PUBLIC GMP::gmp GMP::gmpxx)
  endif()

  # add MQT alias
  add_library(MQT::CoreZX ALIAS ${PROJECT_NAME}-zx)
  add_library(MQT::${OLD_PROJECT_NAME}_zx ALIAS ${PROJECT_NAME}-zx)
endif()
