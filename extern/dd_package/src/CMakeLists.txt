cmake_minimum_required (VERSION 3.10)
project( dd_package_lib
        LANGUAGES CXX
        VERSION 1.1
        DESCRIPTION "Library to handle the decisions diagrams and corresponding operations")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -Wall -Wextra -Wpedantic -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DNDEBUG -g0 -O3")

add_library(${PROJECT_NAME} STATIC DDpackage.cpp DDcomplex.cpp DDpackage.h DDcomplex.h)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
	set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
target_compile_options(${PROJECT_NAME} PRIVATE -mtune=native -march=native)
target_include_directories(${PROJECT_NAME} PUBLIC ./)

