cmake_minimum_required (VERSION 3.2)
project( dd_package
        LANGUAGES CXX
        VERSION 1.1
        DESCRIPTION "Library to handle the decisions diagrams and corresponding operations")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -Wall -Wextra -Wpedantic -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DNDEBUG -g0 -O3")

add_executable(dd_example src/main.cpp src/DDcomplex.cpp src/DDpackage.cpp)

