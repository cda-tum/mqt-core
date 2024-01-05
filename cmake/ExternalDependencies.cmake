# Declare all external dependencies and make sure that they are available.

include(FetchContent)
set(FETCH_PACKAGES "")

# A macro to declare a dependency that takes into account the different CMake versions and the
# features that they make available. In particular: - CMake 3.24 introduced the `FIND_PACKAGE_ARGS`
# option to `FetchContent` which allows to combine `FetchContent_Declare` and `find_package` in a
# single call. - CMake 3.25 introduced the `SYSTEM` option to `FetchContent_Declare` which marks the
# dependency as a system dependency. This is useful to avoid compiler warnings from external header
# only libraries. - CMake 3.28 introduced the `EXCLUDE_FROM_ALL` option to `FetchContent_Declare`
# which allows to exclude all targets from the dependency from the `all` target.
macro(DECLARE_DEPENDENCY)
  cmake_parse_arguments(DEPENDENCY "SYSTEM;EXCLUDE_FROM_ALL" "NAME;URL;MD5;MIN_VERSION;ALT_NAME" ""
                        ${ARGN})
  set(ADDITIONAL_OPTIONS "")
  if(DEPENDENCY_SYSTEM AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
    list(APPEND ADDITIONAL_OPTIONS SYSTEM)
  endif()
  if(DEPENDENCY_EXCLUDE_FROM_ALL AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    list(APPEND ADDITIONAL_OPTIONS EXCLUDE_FROM_ALL)
  endif()
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.28)
    FetchContent_Declare(
      ${DEPENDENCY_NAME}
      URL ${DEPENDENCY_URL}
      URL_MD5 ${DEPENDENCY_MD5}
      ${ADDITIONAL_OPTIONS} FIND_PACKAGE_ARGS ${DEPENDENCY_MIN_VERSION} NAMES
      ${DEPENDENCY_ALT_NAME})
    list(APPEND FETCH_PACKAGES ${DEPENDENCY_NAME})
  elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
    FetchContent_Declare(
      ${DEPENDENCY_NAME}
      URL ${DEPENDENCY_URL}
      URL_MD5 ${DEPENDENCY_MD5}
      ${ADDITIONAL_OPTIONS} FIND_PACKAGE_ARGS ${DEPENDENCY_MIN_VERSION} NAMES
      ${DEPENDENCY_ALT_NAME})
    list(APPEND FETCH_PACKAGES ${DEPENDENCY_NAME})
  elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    FetchContent_Declare(
      ${DEPENDENCY_NAME}
      URL ${DEPENDENCY_URL}
      URL_MD5 ${DEPENDENCY_MD5}
      ${ADDITIONAL_OPTIONS} FIND_PACKAGE_ARGS ${DEPENDENCY_MIN_VERSION} NAMES
      ${DEPENDENCY_ALT_NAME})
    list(APPEND FETCH_PACKAGES ${DEPENDENCY_NAME})
  else()
    # try to get the system installed version
    find_package(${DEPENDENCY_NAME} ${DEPENDENCY_MIN_VERSION} QUIET NAMES ${DEPENDENCY_ALT_NAME})
    if(NOT ${DEPENDENCY_NAME}_FOUND)
      FetchContent_Declare(
        ${DEPENDENCY_NAME}
        URL ${DEPENDENCY_URL}
        URL_MD5 ${DEPENDENCY_MD5})
      list(APPEND FETCH_PACKAGES ${DEPENDENCY_NAME})
    endif()
  endif()
endmacro()

set(JSON_BuildTests
    OFF
    CACHE INTERNAL "")
set(JSON_MultipleHeaders
    OFF
    CACHE INTERNAL "")
declare_dependency(
  NAME
  nlohmann_json
  URL
  https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  MD5
  c23a33f04786d85c29fda8d16b5f0efd
  MIN_VERSION
  3.11.3
  SYSTEM
  EXCLUDE_FROM_ALL)

set(BOOST_MP_STANDALONE
    ON
    CACHE INTERNAL "Use standalone boost multiprecision")
declare_dependency(
  NAME
  boost_multiprecision
  URL
  https://github.com/boostorg/multiprecision/archive/refs/tags/Boost_1_84_0.tar.gz
  MD5
  b829378c90f4b268c79a796025c43eee
  MIN_VERSION
  1.80.0
  ALT_NAME
  Boost
  SYSTEM
  EXCLUDE_FROM_ALL)

if(BUILD_MQT_CORE_TESTS)
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  set(FP_ARGS 1.14.0 NAMES GTest)
  declare_dependency(
    NAME
    googletest
    URL
    https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    MD5
    c8340a482851ef6a3fe618a082304cfc
    MIN_VERSION
    1.14.0
    ALT_NAME
    GTest
    SYSTEM
    EXCLUDE_FROM_ALL)
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

  declare_dependency(
    NAME
    pybind11_json
    URL
    https://github.com/pybind/pybind11_json/archive/refs/tags/0.2.13.tar.gz
    MD5
    93ebbea2bb69f71febe0f83c8f88ced2
    MIN_VERSION
    0.2.13
    SYSTEM
    EXCLUDE_FROM_ALL)
endif()

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
