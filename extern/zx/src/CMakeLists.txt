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

# set required C++ standard and disable compiler specific extensions
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

add_subdirectory("${PROJECT_SOURCE_DIR}/extern/boost/config" "extern/boost/config" EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME} PUBLIC Boost::config)

add_subdirectory("${PROJECT_SOURCE_DIR}/extern/boost/multiprecision" "extern/boost/multiprecision"
                 EXCLUDE_FROM_ALL)
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

# set compiler flags depending on compiler
if(MSVC)
  target_compile_options(${PROJECT_NAME} PUBLIC /utf-8 /W4)
else()
  target_compile_options(
    ${PROJECT_NAME}
    PUBLIC -Wall
           -Wextra
           -pedantic
           -g
           $<$<CONFIG:RELEASE>:-fno-math-errno
           -ffinite-math-only
           -fno-trapping-math>)
  if(BINDINGS AND NOT WIN32)
    # adjust visibility settings for building Python bindings
    target_compile_options(${PROJECT_NAME} PUBLIC -fvisibility=hidden)
  endif()
  if(NOT DEPLOY)
    # only include machine-specific optimizations when building for the host machine
    target_compile_options(${PROJECT_NAME} PUBLIC -mtune=native)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
    if(HAS_MARCH_NATIVE)
      target_compile_options(${PROJECT_NAME} PUBLIC -march=native)
    endif()
  endif()
endif()

# enable interprocedural optimization if it is supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
  set_target_properties(${TARGETNAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(GENERATE_POSITION_INDEPENDENT_CODE OR BINDINGS)
  include(CheckPIESupported)
  check_pie_supported()
  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()

# add coverage compiler and linker flag to the library and all targets that link against it, if
# COVERAGE is set
if(COVERAGE)
  target_compile_options(${PROJECT_NAME} PUBLIC --coverage)
  target_link_libraries(${PROJECT_NAME} PUBLIC --coverage)
endif()

# add MQT alias
add_library(MQT::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
