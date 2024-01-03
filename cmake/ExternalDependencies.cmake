# Declare all external dependencies and make sure that they are available.

include(FetchContent)
set(FETCH_PACKAGES "")

if(BUILD_MQT_CORE_TESTS)
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
      URL_MD5 c8340a482851ef6a3fe618a082304cfc
      EXCLUDE_FROM_ALL SYSTEM FIND_PACKAGE_ARGS NAMES GTest)
    list(APPEND FETCH_PACKAGES googletest)
  else()
    # try to get the system installed version
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
      FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_MD5 c8340a482851ef6a3fe618a082304cfc)
      list(APPEND FETCH_PACKAGES googletest)
    endif()
  endif()
endif()

set(JSON_BuildTests
    OFF
    CACHE INTERNAL "")
set(JSON_MultipleHeaders
    OFF
    CACHE INTERNAL "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_MD5 c23a33f04786d85c29fda8d16b5f0efd
    SYSTEM EXCLUDE_FROM_ALL FIND_PACKAGE_ARGS)
  list(APPEND FETCH_PACKAGES nlohmann_json)
else()
  # try to get the system installed version
  find_package(nlohmann_json QUIET)
  if(NOT nlohmann_json_FOUND)
    FetchContent_Declare(
      nlohmann_json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
      URL_MD5 c23a33f04786d85c29fda8d16b5f0efd)
    list(APPEND FETCH_PACKAGES nlohmann_json)
  endif()
endif()

if(BINDINGS)
  if(NOT SKBUILD)
    # Manually detect the installed pybind11 package and import it into CMake.
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE pybind11_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
  endif()

  # add pybind11 library
  find_package(pybind11 CONFIG REQUIRED)

  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    FetchContent_Declare(
      pybind11_json
      URL https://github.com/pybind/pybind11_json/archive/refs/tags/0.2.13.tar.gz
      URL_MD5 93ebbea2bb69f71febe0f83c8f88ced2
      SYSTEM EXCLUDE_FROM_ALL FIND_PACKAGE_ARGS)
    list(APPEND FETCH_PACKAGES pybind11_json)
  else()
    # try to get the system installed version
    find_package(pybind11_json QUIET)
    if(NOT pybind11_json_FOUND)
      FetchContent_Declare(
        pybind11_json
        URL https://github.com/pybind/pybind11_json/archive/refs/tags/0.2.13.tar.gz
        URL_MD5 93ebbea2bb69f71febe0f83c8f88ced2)
      list(APPEND FETCH_PACKAGES pybind11_json)
    endif()
  endif()
endif()

set(BOOST_MP_STANDALONE
    ON
    CACHE INTERNAL "Use standalone boost multiprecision")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  FetchContent_Declare(
    boost_multiprecision
    URL https://github.com/boostorg/multiprecision/archive/refs/tags/Boost_1_84_0.tar.gz
    URL_MD5 b829378c90f4b268c79a796025c43eee
    SYSTEM EXCLUDE_FROM_ALL FIND_PACKAGE_ARGS 1.74.0 NAMES Boost)
  list(APPEND FETCH_PACKAGES boost_multiprecision)
else()
  # try to get the system installed version
  find_package(Boost 1.74.0 QUIET)
  if(NOT Boost_FOUND)
    FetchContent_Declare(
      boost_multiprecision
      URL https://github.com/boostorg/multiprecision/archive/refs/tags/Boost_1_84_0.tar.gz
      URL_MD5 b829378c90f4b268c79a796025c43eee)
    list(APPEND FETCH_PACKAGES boost_multiprecision)
  endif()
endif()

FetchContent_MakeAvailable(${FETCH_PACKAGES})
