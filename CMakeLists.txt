cmake_minimum_required(VERSION 3.10)
project(QFR
        VERSION 0.9.3
        DESCRIPTION "QFR  - A library for the representation of quantum functionality"
        LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

SET(COVERAGE OFF CACHE BOOL "Coverage")

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
	    STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
	             "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Git QUIET)
option(GIT_SUBMODULE "Check submodules during build" ON)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git" AND GIT_SUBMODULE)
	# Update submodules as needed
	message(STATUS "dd_package update")
	execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive --remote -- extern/dd_package
	                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	                RESULT_VARIABLE GIT_SUBMOD_RESULT)
	if(NOT GIT_SUBMOD_RESULT EQUAL "0")
		message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
	endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/dd_package/CMakeLists.txt")
	message(FATAL_ERROR "DD Package failed to download! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git" AND GIT_SUBMODULE)
		message(STATUS "googletest update")
		execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- extern/googletest
		                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		                RESULT_VARIABLE GIT_SUBMOD_RESULT)
		if(NOT GIT_SUBMOD_RESULT EQUAL "0")
			message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
		endif()
	endif()

	if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/googletest/CMakeLists.txt")
		message(FATAL_ERROR "GoogleTest failed to download! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
	endif()

	enable_testing()
	include(GoogleTest)
	add_subdirectory(test)
endif()

add_subdirectory(apps)
