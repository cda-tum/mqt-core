# set required cmake version
cmake_minimum_required(VERSION 3.24...3.30)

project(
  mqt-mlir
  LANGUAGES C CXX
  DESCRIPTION "MQT MLIR - Interfacing between the Munich Quantum Toolkit and MLIR")

set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "C++ standard to conform to")

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# check if this is the master project or used via add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MQT_MLIR_MASTER_PROJECT ON)
else()
  set(MQT_MLIR_MASTER_PROJECT OFF)
endif()

option(BUILD_MQT_MLIR_TESTS "Also build tests for the MQT MLIR project" ${MQT_MLIR_MASTER_PROJECT})

include(cmake/ExternalDependencies.cmake)

# set the include directory for the build tree
set(MQT_MLIR_INCLUDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# MLIR must be installed on the system
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add the paths to the MLIR and LLVM CMake modules.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include the TableGen, LLVM and MLIR CMake modules.
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${MQT_MLIR_INCLUDE_BUILD_DIR})
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# add main library code
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tools)

# add test code
if(BUILD_MQT_MLIR_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
