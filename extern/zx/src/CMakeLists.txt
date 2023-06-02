if(NOT TARGET ${PROJECT_NAME})
  # main project library
  add_library(
    ${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/include/Rational.hpp
    ${PROJECT_SOURCE_DIR}/include/ZXDiagram.hpp
    ${PROJECT_SOURCE_DIR}/include/Definitions.hpp
    ${PROJECT_SOURCE_DIR}/include/Rules.hpp
    ${PROJECT_SOURCE_DIR}/include/Simplify.hpp
    ${PROJECT_SOURCE_DIR}/include/Utils.hpp
    ${PROJECT_SOURCE_DIR}/include/Expression.hpp
    Rational.cpp
    ZXDiagram.cpp
    Rules.cpp
    Simplify.cpp
    Utils.cpp
    Expression.cpp)

  # set include directories
  target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
                                                    ${PROJECT_BINARY_DIR}/include)

  # add options and warnings to the library
  target_link_libraries(${PROJECT_NAME} PUBLIC project_options project_warnings)

  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/boost/config" "extern/boost/config"
                   EXCLUDE_FROM_ALL)
  target_link_libraries(${PROJECT_NAME} PUBLIC Boost::config)

  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/boost/multiprecision"
                   "extern/boost/multiprecision" EXCLUDE_FROM_ALL)
  target_link_libraries(${PROJECT_NAME} PUBLIC Boost::multiprecision)
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

  # # link to GMP libraries if present
  if(GMP_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GMP)
    target_link_libraries(${PROJECT_NAME} PUBLIC GMP::gmp GMP::gmpxx)
  endif()

  # add MQT alias
  add_library(MQT::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
endif()
