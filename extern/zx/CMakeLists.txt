cmake_minimum_required(VERSION 3.14...3.22)

project(zx
        LANGUAGES CXX
        VERSION 0.1
        DESCRIPTION "ZX - An MQT library for working with ZX-diagrams"
        )

# enable organization of targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# configuration options
option(DEPLOY "Configure for deployment")
option(BINDINGS "Configure for building Python bindings")
option(COVERAGE "Configure for coverage report generation")
option(BUILD_ZX_TESTS "Also build tests for ZX library")

# build type settings
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

macro(check_submodule_present MODULENAME)
	if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${MODULENAME}/CMakeLists.txt")
		message(FATAL_ERROR "${MODULENAME} submodule not cloned properly. Please run `git submodule update --init --recursive` from the main project directory")
	endif()
endmacro()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(GMP)
if(GMP_FOUND)
  add_library(gmp::gmpxx IMPORTED INTERFACE)
  add_library(gmp::libgmpxx IMPORTED INTERFACE)
  add_library(gmp::libgmp IMPORTED INTERFACE)
  add_compile_definitions(GMP)
else()
  # message(WARNING "Did not find GMP. ZX library and other depending target will not be available")
check_submodule_present(config)
check_submodule_present(multiprecision)
set(BOOST_MP_STANDALONE ON)
add_compile_definitions(BOOST_MP_STANDALONE)
endif()


# add main library code
add_subdirectory(src)

# add test code
if(BUILD_ZX_TESTS)
	enable_testing()
	include(GoogleTest)
	add_subdirectory(test)
endif()
