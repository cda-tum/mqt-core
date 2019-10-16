cmake_minimum_required(VERSION 3.0)
project(IntermediateRepresentation)

set(CMAKE_CXX_STANDARD 14)

# Set a default build type for single-configuration
# CMake generators if no build type is set.
IF (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -g -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-Og")

if (${CMAKE_CXX_COMPILER_ID} MATCHES "Intel")
	set(${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS} -shared-intel -debug inline-debug-info)
endif ()

add_subdirectory(src)
add_subdirectory(test)
