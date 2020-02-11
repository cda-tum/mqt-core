cmake_minimum_required (VERSION 3.10)
project( dd_package_lib
        LANGUAGES CXX
        VERSION 1.1
        DESCRIPTION "Library to handle the decisions diagrams and corresponding operations")

add_library(${PROJECT_NAME} STATIC DDpackage.cpp DDcomplex.cpp DDpackage.h DDcomplex.h)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_EXTENSIONS OFF)
set_property(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
	set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2 /GL>")
	target_link_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -DNDEBUG -g0 $<$<CONFIG:DEBUG>:-Og -Wextra -Wpedantic -pedantic> $<$<CONFIG:RELEASE>:-O3 -mtune=native -march=native>)
endif()

target_compile_options(${PROJECT_NAME} PRIVATE )
target_include_directories(${PROJECT_NAME} PUBLIC ./)
