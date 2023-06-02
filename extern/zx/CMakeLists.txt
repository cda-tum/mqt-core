project(
  zx
  LANGUAGES CXX
  VERSION 0.1
  DESCRIPTION "ZX - An MQT library for working with ZX-diagrams")

check_submodule_present(boost/config)
check_submodule_present(boost/multiprecision)
set(BOOST_MP_STANDALONE ON)

find_package(GMP)
if(NOT GMP_FOUND)
  message(NOTICE "Did not find GMP. Using Boost multiprecision library instead.")
endif()

# add main library code
add_subdirectory(src)

# add test code
option(BUILD_ZX_TESTS "Also build tests for ZX library")
if(BUILD_ZX_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
