# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Declare all external dependencies and make sure that they are available.

include(FetchContent)
include(CMakeDependentOption)
set(FETCH_PACKAGES "")

if(BUILD_MQT_CORE_BINDINGS)
  if(NOT SKBUILD)
    # Manually detect the installed pybind11 package and import it into CMake.
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE pybind11_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
  endif()

  message(STATUS "Python executable: ${Python_EXECUTABLE}")

  # add pybind11 library
  find_package(pybind11 2.13.5 CONFIG REQUIRED)
endif()

set(JSON_VERSION
    3.11.3
    CACHE STRING "nlohmann_json version")
set(JSON_URL https://github.com/nlohmann/json/releases/download/v${JSON_VERSION}/json.tar.xz)
set(JSON_SystemInclude
    ON
    CACHE INTERNAL "Treat the library headers like system headers")
cmake_dependent_option(JSON_Install "Install nlohmann_json library" ON "MQT_CORE_INSTALL" OFF)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(nlohmann_json URL ${JSON_URL} FIND_PACKAGE_ARGS ${JSON_VERSION})
  list(APPEND FETCH_PACKAGES nlohmann_json)
else()
  find_package(nlohmann_json ${JSON_VERSION} QUIET)
  if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(nlohmann_json URL ${JSON_URL})
    list(APPEND FETCH_PACKAGES nlohmann_json)
  endif()
endif()

option(USE_SYSTEM_BOOST "Whether to try to use the system Boost installation" OFF)
set(BOOST_MIN_VERSION
    1.80.0
    CACHE STRING "Minimum required Boost version")
if(USE_SYSTEM_BOOST)
  find_package(Boost ${BOOST_MIN_VERSION} CONFIG REQUIRED)
else()
  set(BOOST_MP_STANDALONE
      ON
      CACHE INTERNAL "Use standalone boost multiprecision")
  set(BOOST_VERSION
      1_86_0
      CACHE INTERNAL "Boost version")
  set(BOOST_URL
      https://github.com/boostorg/multiprecision/archive/refs/tags/Boost_${BOOST_VERSION}.tar.gz)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    FetchContent_Declare(boost_mp URL ${BOOST_URL} FIND_PACKAGE_ARGS ${BOOST_MIN_VERSION} CONFIG
                                      NAMES boost_multiprecision)
    list(APPEND FETCH_PACKAGES boost_mp)
  else()
    find_package(boost_mp ${BOOST_MIN_VERSION} QUIET CONFIG NAMES boost_multiprecision)
    if(NOT boost_mp_FOUND)
      FetchContent_Declare(boost_mp URL ${BOOST_URL})
      list(APPEND FETCH_PACKAGES boost_mp)
    endif()
  endif()
endif()

if(BUILD_MQT_CORE_TESTS)
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  set(GTEST_VERSION
      1.16.0
      CACHE STRING "Google Test version")
  set(GTEST_URL https://github.com/google/googletest/archive/refs/tags/v${GTEST_VERSION}.tar.gz)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    FetchContent_Declare(googletest URL ${GTEST_URL} FIND_PACKAGE_ARGS ${GTEST_VERSION} NAMES GTest)
    list(APPEND FETCH_PACKAGES googletest)
  else()
    find_package(googletest ${GTEST_VERSION} QUIET NAMES GTest)
    if(NOT googletest_FOUND)
      FetchContent_Declare(googletest URL ${GTEST_URL})
      list(APPEND FETCH_PACKAGES googletest)
    endif()
  endif()
endif()

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
