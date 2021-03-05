# set required cmake version
cmake_minimum_required(VERSION 3.14...3.19)

# project definition
project( dd_package
         LANGUAGES CXX
         VERSION 1.2.2
         DESCRIPTION "JKQ decision diagram package tailored to quantum computing")

# enable organization of targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# configuration options
option(DEPLOY "Configure for deployment")
option(BINDINGS "Configure for building Python bindings")
option(COVERAGE "Configure for coverage report generation")
option(GENERATE_POSITION_INDEPENDENT_CODE "Generate position independent code")
option(BUILD_DD_PACKAGE_TESTS "Also build tests for DD package" )

# build type settings
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# add main library code
add_subdirectory(src)

# add test code
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR BUILD_DD_PACKAGE_TESTS)
	if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/googletest/CMakeLists.txt")
		message(FATAL_ERROR "GoogleTest submodule not cloned properly. Please run `git submodule update --init --recursive` from the main project directory")
	endif()

	enable_testing()
	include(GoogleTest)
	add_subdirectory(test)
endif()
