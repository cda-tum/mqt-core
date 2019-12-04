cmake_minimum_required (VERSION 3.2)
project( dd_package_lib
        LANGUAGES CXX
        VERSION 1.1
        DESCRIPTION "Library to handle the decisions diagrams and corresponding operations")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -Wall -Wextra -Wpedantic -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DNDEBUG -g0 -O3")

add_library(dd_package_lib STATIC DDpackage.cpp DDcomplex.cpp)
set_property(TARGET dd_package_lib PROPERTY CXX_STANDARD 11)
set_property(TARGET dd_package_lib PROPERTY CXX_EXTENSIONS OFF)
target_compile_options(dd_package_lib PRIVATE -mtune=native -march=native -flto)
target_include_directories(dd_package_lib PUBLIC ./)

