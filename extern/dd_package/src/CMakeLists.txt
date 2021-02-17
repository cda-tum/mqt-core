# main project library
add_library(${PROJECT_NAME}
            ${CMAKE_CURRENT_SOURCE_DIR}/DDpackage.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DDdebug.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DDcomplex.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/DDexport.cpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/DDpackage.h
            ${${PROJECT_NAME}_SOURCE_DIR}/include/DDcomplex.h
            ${${PROJECT_NAME}_SOURCE_DIR}/include/DDexport.h)

# set include directories
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>)

# set required C++ standard and disable compiler specific extensions
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# enable interprocedural optimization if it is supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
	set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# set compiler flags depending on compiler
if(MSVC)
	target_compile_options(${PROJECT_NAME} PUBLIC /utf-8)
else()
	target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra $<$<CONFIG:DEBUG>:-Og>)
	if (BINDINGS)
		# adjust visibility settings for building Python bindings
		target_compile_options(${PROJECT_NAME} PUBLIC -fvisibility=hidden)
	endif()
	if(NOT DEPLOY)
		# only include machine-specific optimizations when building for the host machine
		target_compile_options(${PROJECT_NAME} PUBLIC -mtune=native -march=native)
	endif ()
endif()

if(GENERATE_POSITION_INDEPENDENT_CODE OR BINDINGS)
	include(CheckPIESupported)
	check_pie_supported()
	set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()

# add coverage compiler and linker flag to the library and all targets that link against it, if COVERAGE is set
if (COVERAGE)
	target_compile_options(${PROJECT_NAME} PUBLIC --coverage)
	target_link_libraries(${PROJECT_NAME} PUBLIC --coverage)
endif()

# add JKQ alias
add_library(JKQ::DDpackage ALIAS ${PROJECT_NAME})
