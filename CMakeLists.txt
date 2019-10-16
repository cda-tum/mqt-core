cmake_minimum_required(VERSION 3.0)
project(IntermediateRepresentation)

# Set a default build type for single-configuration
# CMake generators if no build type is set.
IF (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

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
add_subdirectory(test)
