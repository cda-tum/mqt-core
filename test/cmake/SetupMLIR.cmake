# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.28)
if(NOT MQT_SOURCE_DIR OR NOT TEST_DIR)
  message(FATAL_ERROR "MQT_SOURCE_DIR and TEST_DIR are required")
endif()
include(CMakePackageConfigHelpers)
file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}/source" "${TEST_DIR}/installer/installation")

# A rejected package must never import targets or change LLVM settings.
file(WRITE "${TEST_DIR}/old/MLIRConfig.cmake"
     "message(FATAL_ERROR \"An incompatible MLIR package was imported\")\n")
write_basic_package_version_file(
  "${TEST_DIR}/old/MLIRConfigVersion.cmake"
  VERSION 22.1.8
  COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
foreach(package LLVM MLIR)
  string(TOLOWER "${package}" directory)
  write_basic_package_version_file(
    "${TEST_DIR}/sdk/lib/cmake/${directory}/${package}ConfigVersion.cmake"
    VERSION 23.1.2
    COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
endforeach()
file(
  WRITE "${TEST_DIR}/sdk/lib/cmake/mlir/MLIRConfig.cmake"
  [=[
find_package(LLVM 23.1 REQUIRED CONFIG)
set(MLIR_VERSION 23.1.2)
set(MLIR_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
]=])
file(
  WRITE "${TEST_DIR}/sdk/lib/cmake/llvm/LLVMConfig.cmake"
  [=[
set(LLVM_PACKAGE_VERSION 23.1.2)
set(LLVM_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
include("${CMAKE_CURRENT_LIST_DIR}/assertions.cmake")
]=])
file(WRITE "${TEST_DIR}/sdk/lib/cmake/llvm/assertions.cmake" "set(LLVM_ENABLE_ASSERTIONS ON)\n")
foreach(module TableGen AddLLVM AddMLIR HandleLLVMOptions)
  file(WRITE "${TEST_DIR}/sdk/lib/cmake/llvm/${module}.cmake" "")
endforeach()

# Replace only the network installer; package discovery and SetupMLIR run unchanged.
set(install_script
    [=[
file(COPY "@TEST_DIR@/sdk/" DESTINATION "${INSTALL_PREFIX}")
file(WRITE "${INSTALL_PREFIX}/lib/cmake/llvm/assertions.cmake"
     "set(LLVM_ENABLE_ASSERTIONS ${ASSERTIONS})\n")
file(APPEND "@TEST_DIR@/installs" "${ASSERTIONS}\n")
]=])
string(CONFIGURE "${install_script}" install_script @ONLY)
file(WRITE "${TEST_DIR}/installer/install.cmake" "${install_script}")
set(bash_script
    [=[
set -eu
while getopts "v:p:a:" opt; do
  case "$opt" in v) version="$OPTARG";; p) prefix="$OPTARG";; a) assertions="$OPTARG";; esac
done
test "$version" = 23.1.2
"@CMAKE_COMMAND@" "-DINSTALL_PREFIX=$prefix" "-DASSERTIONS=$assertions" -P "@TEST_DIR@/installer/install.cmake"
]=])
string(CONFIGURE "${bash_script}" bash_script @ONLY)
file(WRITE "${TEST_DIR}/installer/installation/setup-mlir.sh" "${bash_script}")
set(powershell_script
    [=[
param([string]$llvm_version, [string]$install_prefix, [switch]$no_assertions)
if ($llvm_version -ne "23.1.2") { throw "Unexpected SDK version: $llvm_version" }
$assertions = if ($no_assertions) { "OFF" } else { "ON" }
& "@CMAKE_COMMAND@" "-DINSTALL_PREFIX=$install_prefix" "-DASSERTIONS=$assertions" -P "@TEST_DIR@/installer/install.cmake"
exit $LASTEXITCODE
]=])
string(CONFIGURE "${powershell_script}" powershell_script @ONLY)
file(WRITE "${TEST_DIR}/installer/installation/setup-mlir.ps1" "${powershell_script}")
set(project
    [=[
cmake_minimum_required(VERSION 3.28)
project(SetupMLIRTest LANGUAGES NONE)
set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY FALSE)
set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_ROOT_PATH "@TEST_DIR@")
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
if(PARENT_SDK)
  set(LLVM_VERSION_MAJOR "${PARENT_SDK}")
  set(LLVM_VERSION_MINOR 1)
  set(LLVM_PACKAGE_VERSION "${PARENT_SDK}.1.2")
  add_library(LLVMSupport INTERFACE IMPORTED)
endif()
include("@MQT_SOURCE_DIR@/cmake/SetupMLIR.cmake")
if(NOT LLVM_ENABLE_ASSERTIONS STREQUAL EXPECTED_ASSERTIONS)
  message(FATAL_ERROR "Wrong LLVM assertion variant: ${LLVM_ENABLE_ASSERTIONS}")
endif()
if(NOT MLIR_DIR STREQUAL "${EXPECTED_PREFIX}/lib/cmake/mlir"
   OR NOT LLVM_DIR STREQUAL "${EXPECTED_PREFIX}/lib/cmake/llvm")
  message(FATAL_ERROR "LLVM and MLIR must come from the expected SDK")
endif()
]=])
string(CONFIGURE "${project}" project @ONLY)
file(WRITE "${TEST_DIR}/source/CMakeLists.txt" "${project}")
set(configure "${CMAKE_COMMAND}" -S "${TEST_DIR}/source"
              "-DFETCHCONTENT_SOURCE_DIR_SETUP_MLIR=${TEST_DIR}/installer")

# Reconfigure once, then switch variants in the same build directory.
set(old_package "-DMLIR_DIR=${TEST_DIR}/old")
foreach(step ON REUSE OFF)
  set(assertions "${step}")
  set(disconnected OFF)
  set(auto_install ON)
  if(step STREQUAL REUSE)
    set(assertions ON)
    set(disconnected ON)
    set(auto_install OFF)
  endif()
  execute_process(
    COMMAND
      ${configure} -B "${TEST_DIR}/build" ${old_package} "-DMQT_CORE_MLIR_ASSERTIONS=${assertions}"
      "-DMQT_CORE_AUTO_INSTALL_MLIR=${auto_install}"
      "-DFETCHCONTENT_FULLY_DISCONNECTED=${disconnected}" "-DEXPECTED_ASSERTIONS=${assertions}"
      "-DEXPECTED_PREFIX=${TEST_DIR}/build/_deps/llvm-23.1.2-${assertions}" COMMAND_ERROR_IS_FATAL
      ANY)
  unset(old_package)
endforeach()
execute_process(
  COMMAND
    ${configure} -B "${TEST_DIR}/supplied" -DMQT_CORE_AUTO_INSTALL_MLIR=OFF -DPARENT_SDK=23
    "-DMLIR_DIR=${TEST_DIR}/sdk/lib/cmake/mlir" "-DLLVM_DIR=${TEST_DIR}/sdk/lib/cmake/llvm"
    -DEXPECTED_ASSERTIONS=ON "-DEXPECTED_PREFIX=${TEST_DIR}/sdk" COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND
    ${configure} -B "${TEST_DIR}/parent" -DPARENT_SDK=22 "-DMLIR_DIR=${TEST_DIR}/sdk/lib/cmake/mlir"
    "-DLLVM_DIR=${TEST_DIR}/sdk/lib/cmake/llvm" -DEXPECTED_ASSERTIONS=ON
    "-DEXPECTED_PREFIX=${TEST_DIR}/sdk"
  RESULT_VARIABLE parent_result
  OUTPUT_VARIABLE parent_output
  ERROR_VARIABLE parent_error)
if(parent_result EQUAL 0 OR NOT "${parent_output}${parent_error}" MATCHES "embedding project")
  message(
    FATAL_ERROR
      "An incompatible embedding project must be rejected: ${parent_output}${parent_error}")
endif()
file(READ "${TEST_DIR}/installs" installs)
if(NOT installs STREQUAL "ON\nOFF\n")
  message(FATAL_ERROR "Expected one installation per variant, got: ${installs}")
endif()
