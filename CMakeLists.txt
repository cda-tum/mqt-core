# set required cmake version
cmake_minimum_required(VERSION 3.19)

project(
  qfr
  LANGUAGES CXX
  DESCRIPTION "MQT QFR  - A library for Quantum Functionality Representation")

# check whether the submodule ``modulename`` is correctly cloned in the ``/extern`` directory.
macro(CHECK_SUBMODULE_PRESENT modulename)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${modulename}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "${modulename} submodule not cloned properly. \
        Please run `git submodule update --init --recursive` \
        from the main project directory")
  endif()
endmacro()

check_submodule_present(dd_package)
check_submodule_present(json)
check_submodule_present(pybind11)
check_submodule_present(pybind11_json)
check_submodule_present(zx)

# add main library code
add_subdirectory(src)

# add test code
if(BUILD_QFR_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()

# add Python binding code
if(BINDINGS)
  add_subdirectory(mqt/qfr)
endif()
