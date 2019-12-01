cmake_minimum_required(VERSION 3.10)
project(QuantumFunctionalityRepresentation
        VERSION 0.1
        DESCRIPTION "A library for the representation of quantum functionality"
        LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

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
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
	# Update submodules as needed
	option(GIT_SUBMODULE "Check submodules during build" ON)
	if(GIT_SUBMODULE)
		message(STATUS "Submodule update")
		execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
		                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		                RESULT_VARIABLE GIT_SUBMOD_RESULT)
		if(NOT GIT_SUBMOD_RESULT EQUAL "0")
			message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
		endif()
	endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/dd_package/CMakeLists.txt")
	message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

<<<<<<< HEAD
<<<<<<< HEAD
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE  "-g0 -march=native -DNDEBUG-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -Og")
=======
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -g -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3")
<<<<<<< HEAD
set(CMAKE_CXX_FLAGS_DEBUG "-Og -fno-omit-frame-pointer")
>>>>>>> d120d89... Fixed caching issues in transpose and trace.
=======
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -fno-omit-frame-pointer -fno-inline-small-functions")
>>>>>>> 5a096f3... Smaller modifications.master Fixed Complex Table polution by Trace.

if (${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
	set(${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS} -shared-intel -debug inline-debug-info)
endif ()

=======
>>>>>>> e9970ae... DD package fixes
add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	enable_testing()
	include(GoogleTest)
	add_subdirectory(test)
endif()
