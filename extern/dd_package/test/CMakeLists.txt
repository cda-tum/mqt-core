# fix cmake warnings for googletest when building shared libs
if (APPLE AND BUILD_SHARED_LIBS)
	set(CMAKE_MACOSX_RPATH ON)
endif ()
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest" EXCLUDE_FROM_ALL)
mark_as_advanced(
		BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
		gmock_build_tests gtest_build_samples gtest_build_tests
		gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)
set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)
target_compile_features(gtest PUBLIC cxx_std_14)
target_compile_features(gtest_main PUBLIC cxx_std_14)

add_executable(${PROJECT_NAME}_example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME}_example PRIVATE ${PROJECT_NAME})
target_compile_features(${PROJECT_NAME}_example PUBLIC cxx_std_11)
set_property(TARGET ${PROJECT_NAME}_example PROPERTY CXX_EXTENSIONS OFF)
if (COVERAGE)
	target_compile_options(${PROJECT_NAME}_example PRIVATE --coverage)
	target_link_libraries(${PROJECT_NAME}_example PRIVATE --coverage)
endif()
if(MSVC)
	target_compile_options(${PROJECT_NAME}_example PRIVATE "/W4" "/utf-8" "$<$<CONFIG:RELEASE>:/O2 /GL>")
	target_link_options(${PROJECT_NAME}_example PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
	target_compile_options(${PROJECT_NAME}_example PRIVATE -Wall -DNDEBUG -g0 $<$<CONFIG:DEBUG>:-Og -Wextra -Wpedantic -pedantic> $<$<CONFIG:RELEASE>:-O3 -mtune=native -march=native>)
endif()

macro(package_add_test TESTNAME)
	# create an exectuable in which the tests will be stored
	add_executable(${TESTNAME} ${ARGN})
	# link the Google test infrastructure, mocking library, and a default main fuction to
	# the test executable.  Remove g_test_main if writing your own main function.
	target_link_libraries(${TESTNAME} PRIVATE ${PROJECT_NAME} gtest gtest_main)
	target_compile_features(${TESTNAME} PUBLIC cxx_std_14)
	set_property(TARGET ${TESTNAME} PROPERTY CXX_EXTENSIONS OFF)
	if (COVERAGE)
		target_compile_options(${TESTNAME} PRIVATE --coverage)
		target_link_libraries(${TESTNAME} PRIVATE --coverage)
	endif()
	if(MSVC)
		target_compile_options(${TESTNAME} PUBLIC "/W4" "/utf-8" "$<$<CONFIG:RELEASE>:/O2 /GL>")
		target_link_options(${TESTNAME} PUBLIC "$<$<CONFIG:Release>:/LTCG>")
	else()
		target_compile_options(${TESTNAME} PUBLIC -Wall -DNDEBUG -g0 $<$<CONFIG:DEBUG>:-Og -Wextra -Wpedantic -pedantic> $<$<CONFIG:RELEASE>:-O3 -mtune=native -march=native>)
	endif()

	# gtest_discover_tests replaces gtest_add_tests,
	# see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
	gtest_discover_tests(${TESTNAME}
	                     # set a working directory so your project root so that you can find test data via paths relative to the project root
	                     WORKING_DIRECTORY ${PROJECT_DIR}
	                     PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
	                     )
	set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

# add unit tests
package_add_test(${PROJECT_NAME}_test ${CMAKE_CURRENT_SOURCE_DIR}/test_complex.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test_package.cpp)

# include dd_example in test suite
add_test(NAME ${PROJECT_NAME}_example COMMAND ${CMAKE_CURRENT_BINARY_DIR}/dd_package_example)
