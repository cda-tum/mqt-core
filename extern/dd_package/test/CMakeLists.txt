if (NOT TARGET gtest OR NOT TARGET gmock)
	# Prevent overriding the parent project's compiler/linker settings on Windows
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest" EXCLUDE_FROM_ALL)
	mark_as_advanced(
			BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
			gmock_build_tests gtest_build_samples gtest_build_tests
			gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
	)
	set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER extern)
	if (BINDINGS)
		# adjust visibility settings for building Python bindings
		target_compile_options(gtest PUBLIC -fvisibility=hidden)
		target_compile_options(gmock PUBLIC -fvisibility=hidden)
	endif()
endif()

if (NOT TARGET benchmark::benchmark)
	set(THREADS_PREFER_PTHREAD_FLAG ON)
	find_package(Threads REQUIRED)
	set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Skip google benchmark self tests")
	add_subdirectory("${PROJECT_SOURCE_DIR}/extern/benchmark" "extern/benchmark" EXCLUDE_FROM_ALL)
	set_target_properties(benchmark benchmark_main PROPERTIES FOLDER extern)
	if (BINDINGS)
		# adjust visibility settings for building Python bindings
		target_compile_options(benchmark::benchmark PUBLIC -fvisibility=hidden)
	endif()
endif()

add_executable(${PROJECT_NAME}_example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME}_example PRIVATE ${PROJECT_NAME})

macro(package_add_test TESTNAME)
	# create an exectuable in which the tests will be stored
	add_executable(${TESTNAME} ${ARGN})
	# link the Google test infrastructure and a default main fuction to the test executable.
	target_link_libraries(${TESTNAME} PRIVATE ${PROJECT_NAME} gtest_main)
	# discover tests
	gtest_discover_tests(${TESTNAME} WORKING_DIRECTORY ${PROJECT_DIR} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}")
	set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

# add unit tests
package_add_test(${PROJECT_NAME}_test ${CMAKE_CURRENT_SOURCE_DIR}/test_complex.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test_package.cpp)

# include dd_example in test suite
add_test(NAME ${PROJECT_NAME}_example COMMAND ${CMAKE_CURRENT_BINARY_DIR}/dd_package_example)

# add benchmark executable
add_executable(${PROJECT_NAME}_bench ${CMAKE_CURRENT_SOURCE_DIR}/bench_package.cpp)
# link the Google benchmark infrastructure and a default main fuction to the benchmark executable.
target_link_libraries(${PROJECT_NAME}_bench PRIVATE ${PROJECT_NAME} benchmark::benchmark_main Threads::Threads)
set_target_properties(${PROJECT_NAME}_bench PROPERTIES FOLDER tests)
