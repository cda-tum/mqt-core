# main project library
add_library(${PROJECT_NAME}
            ${${PROJECT_NAME}_SOURCE_DIR}/include/Rational.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/ZXDiagram.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/Definitions.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/Rules.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/Simplify.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/Utils.hpp
            ${${PROJECT_NAME}_SOURCE_DIR}/include/Expression.hpp

            ${CMAKE_CURRENT_SOURCE_DIR}/Rational.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/ZXDiagram.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Rules.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Simplify.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/Expression.cpp)

# set include directories
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>)

# set required C++ standard and disable compiler specific extensions
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# link to GMP libraries
target_link_libraries(${PROJECT_NAME} PUBLIC gmpxx gmp::libgmpxx gmp gmp::libgmp)

# set compiler flags depending on compiler
if(MSVC)
	target_compile_options(${PROJECT_NAME} PUBLIC /utf-8 /W4)
else()
	target_compile_options(${PROJECT_NAME} INTERFACE -Wall -Wextra -pedantic $<$<CONFIG:DEBUG>:-Og>$<$<CONFIG:RELEASE>:-fno-math-errno -ffinite-math-only -fno-trapping-math>) # if (BINDINGS AND NOT WIN32)
	if (NOT DEPLOY)
		# only include machine-specific optimizations when building for the host machine
		target_compile_options(${PROJECT_NAME} INTERFACE -mtune=native)
		include(CheckCXXCompilerFlag)
		check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
		if (HAS_MARCH_NATIVE)
			target_compile_options(${PROJECT_NAME} INTERFACE -march=native)
		endif ()
	endif ()
endif ()

# enable interprocedural optimization if it is supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if (ipo_supported)
	set_target_properties(${TARGETNAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

# add coverage compiler and linker flag to the library and all targets that link against it, if COVERAGE is set
if (COVERAGE)
	target_compile_options(${PROJECT_NAME} PUBLIC --coverage)
	target_link_libraries(${PROJECT_NAME} PUBLIC --coverage)
endif ()

# add MQT alias
add_library(MQT::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
