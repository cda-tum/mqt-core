if (NOT TARGET gtest OR NOT TARGET gmock)
	# Prevent overriding the parent project's compiler/linker settings on Windows
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	add_subdirectory("${PROJECT_SOURCE_DIR}/extern/dd_package/extern/googletest" "extern/dd_package/extern/googletest" EXCLUDE_FROM_ALL)
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
	endif ()
endif ()

add_executable(${PROJECT_NAME}_example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME}_example PRIVATE ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_example PROPERTIES FOLDER tests CMAKE_CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
# enable interprocedural optimization if it is supported
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if (ipo_supported)
	set_target_properties(${PROJECT_NAME}_example PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif ()

add_custom_command(TARGET ${PROJECT_NAME}_example
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_DIR:${PROJECT_NAME}_example>/${PROJECT_NAME}_example ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_example
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/circuits/ $<TARGET_FILE_DIR:${PROJECT_NAME}_example>/circuits
                   COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_DIR:${PROJECT_NAME}_example>/circuits ${CMAKE_BINARY_DIR}/circuits
                   COMMENT "Copying circuits and creating symlinks for ${PROJECT_NAME}_example"
                   VERBATIM)

macro(package_add_test TESTNAME)
	# create an exectuable in which the tests will be stored
	add_executable(${TESTNAME} ${ARGN})
	# link the Google test infrastructure and a default main fuction to the test executable.
	target_link_libraries(${TESTNAME} PRIVATE ${PROJECT_NAME} gmock gtest_main)
	# discover tests
	gtest_discover_tests(${TESTNAME} WORKING_DIRECTORY ${PROJECT_DIR} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}")
	set_target_properties(${TESTNAME} PROPERTIES FOLDER tests CMAKE_CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
	# enable interprocedural optimization if it is supported
	include(CheckIPOSupported)
	check_ipo_supported(RESULT ipo_supported)
	if (ipo_supported)
		set_target_properties(${TESTNAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
	endif ()
endmacro()

# add unit tests
package_add_test(${PROJECT_NAME}_test ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_qft.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_grover.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_bernsteinvazirani.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_entanglement.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_grcs.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_random_clifford.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_io.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_ddfunctionality.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_qfr_functionality.cpp)

add_custom_command(TARGET ${PROJECT_NAME}_test
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_DIR:${PROJECT_NAME}_test>/${PROJECT_NAME}_test ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_test
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/circuits $<TARGET_FILE_DIR:${PROJECT_NAME}_test>/circuits
                   COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_DIR:${PROJECT_NAME}_test>/circuits ${CMAKE_BINARY_DIR}/circuits
                   COMMENT "Copying circuits and creating symlinks for ${PROJECT_NAME}_test"
                   VERBATIM)

if (CMAKE_BUILD_TYPE MATCHES pgotrain)

	if (CMAKE_COMPILER_IS_GNUCXX)
		SET(PGO_COMPILE_FLAGS "-fprofile-generate=${CMAKE_BINARY_DIR}/profile-data")
	endif ()
	if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
		SET(PGO_COMPILE_FLAGS "-fprofile-instr-generate -g -O0  -fprofile-arcs -ftest-coverage")
	endif ()

	# Add the CMAKE_CXX_FLAGS_RELEASE so that a PGO optimized build also includes release flags
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${PGO_COMPILE_FLAGS}")

endif ()

if (CMAKE_BUILD_TYPE MATCHES pgobuild)
	# Where to find the profiling data from the training run
	if (NOT PGO_TRAINING_DIR)
		SET(PGO_TRAINING_DIR ../training)
	endif ()
	SET(PGO_TRAINING_DATA ${CMAKE_BINARY_DIR}/${PGO_TRAINING_DIR}/profile-data)

	if (NOT EXISTS ${PGO_TRAINING_DATA})
		message(FATAL_ERROR "No profiling Data Found so can't Build. Ensure that the training run was executed in the training build directory. Training data expected in Directory: " ${PGO_TRAINING_DATA})
	endif ()

	if (CMAKE_COMPILER_IS_GNUCXX)
		SET(PGO_COMPILE_FLAGS "-fprofile-use=${PGO_TRAINING_DATA} -fprofile-correction")
	endif ()
	if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
		SET(PGO_COMPILE_FLAGS "-fprofile-instr-use -g -O0  -fprofile-arcs -ftest-coverage")
	endif ()

	# This custom target always runs.
	# It launches 2 commands which will run in the directory specified by PGO_TRAINING_DIR.
	# First, it runs 'make all' in the Profile instrumented build area
	# Next it runs 'make test' to ensure that the profiling information is generated.
	# In this way, running 'make all' in the final build area guarantees that the Profile
	# instrumented training files will be re-compiled, then the test suite will be run
	# to generate new profiling files, before the final build version is compiled using
	# this profiling information.
	add_custom_target(run_training
	                  ALL
	                  WORKING_DIRECTORY ${PGO_TRAINING_DIR}
	                  COMMAND ${CMAKE_BUILD_TOOL} all
	                  COMMAND ${CMAKE_BUILD_TOOL} ${PROJECT_NAME}_test
	                  VERBATIM)

	# Add the CMAKE_CXX_FLAGS_RELEASE so that a PGO optimized build also includes release flags
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${PGO_COMPILE_FLAGS}")
endif ()
