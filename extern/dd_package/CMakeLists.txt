cmake_minimum_required(VERSION 3.2)
project(dd_package)

# Set a default build type for single-configuration
# CMake generators if no build type is set.
IF (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

add_subdirectory(src)
add_subdirectory(test)
