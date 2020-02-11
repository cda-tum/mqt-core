cmake_minimum_required (VERSION 3.10)
project(dd_package_test)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                RESULT_VARIABLE result
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
	message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif ()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
                RESULT_VARIABLE result
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download)
if (result)
	message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif ()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
	include_directories("${gtest_SOURCE_DIR}/include")
endif ()

add_executable(dd_test test_complex.cpp test_package.cpp)
target_link_libraries(dd_test dd_package_lib gtest gtest_main)
target_compile_features(dd_test PUBLIC cxx_std_11)
set_property(TARGET dd_test PROPERTY CXX_EXTENSIONS OFF)
if(MSVC)
	target_compile_options(dd_test PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2 /GL>")
	target_link_options(dd_test PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
	target_compile_options(dd_test PRIVATE -Wall -DNDEBUG -g0 $<$<CONFIG:DEBUG>:-Og -Wextra -Wpedantic -pedantic> $<$<CONFIG:RELEASE>:-O3 -mtune=native -march=native>)
endif()

add_executable(dd_example main.cpp)
target_link_libraries(dd_example dd_package_lib)
target_compile_features(dd_example PUBLIC cxx_std_11)
set_property(TARGET dd_example PROPERTY CXX_EXTENSIONS OFF)
if(MSVC)
	target_compile_options(dd_example PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2 /GL>")
	target_link_options(dd_example PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
	target_compile_options(dd_example PRIVATE -Wall -DNDEBUG -g0 $<$<CONFIG:DEBUG>:-Og -Wextra -Wpedantic -pedantic> $<$<CONFIG:RELEASE>:-O3 -mtune=native -march=native>)
endif()

enable_testing()
add_test(NAME dd_test COMMAND ./dd_test)
add_test(NAME dd_example COMMAND ./dd_example)
