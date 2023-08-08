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

  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/boost/config" "extern/boost/config"
                   EXCLUDE_FROM_ALL)
  target_link_libraries(${PROJECT_NAME}-zx PUBLIC Boost::config)

  set(BOOST_MP_STANDALONE ON)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/boost/multiprecision"
                   "extern/boost/multiprecision" EXCLUDE_FROM_ALL)
  target_link_libraries(${PROJECT_NAME}-zx PUBLIC Boost::multiprecision)
  # the following sets the SYSTEM flag for the include dirs of the boost libs to suppress warnings
  # cmake-lint: disable=C0307
  set_target_properties(
    boost_config PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                            $<TARGET_PROPERTY:boost_config,INTERFACE_INCLUDE_DIRECTORIES>)
  # cmake-lint: disable=C0307
  set_target_properties(
    boost_multiprecision
    PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
               $<TARGET_PROPERTY:boost_multiprecision,INTERFACE_INCLUDE_DIRECTORIES>)

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
