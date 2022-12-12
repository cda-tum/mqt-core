if(NOT TARGET gtest OR NOT TARGET gmock)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest" EXCLUDE_FROM_ALL)
  mark_as_advanced(
    BUILD_GMOCK
    BUILD_GTEST
    BUILD_SHARED_LIBS
    gmock_build_tests
    gtest_build_samples
    gtest_build_tests
    gtest_disable_pthreads
    gtest_force_shared_crt
    gtest_hide_internal_symbols)
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER extern)
  if(BINDINGS AND NOT WIN32)
    # adjust visibility settings for building Python bindings
    target_compile_options(gtest PUBLIC -fvisibility=hidden)
    target_compile_options(gmock PUBLIC -fvisibility=hidden)
  endif()
endif()

# macro to add a test executable for one of the project libraries
macro(PACKAGE_ADD_TEST testname)
  # create an executable in which the tests will be stored
  add_executable(${testname} ${ARGN})
  # link the Google test infrastructure and a default main function to the test executable.
  target_link_libraries(${testname} PRIVATE ${PROJECT_NAME} gmock gtest_main)
  # discover tests
  gtest_discover_tests(
    ${testname}
    WORKING_DIRECTORY ${PROJECT_DIR}
    PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}")
  set_target_properties(
    ${testname}
    PROPERTIES FOLDER tests
               CMAKE_CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)
  add_custom_command(
    TARGET ${testname}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_DIR:${testname}>/${testname}
            ${CMAKE_BINARY_DIR}/${testname}
    COMMENT "Creating symlinks for ${testname}"
    VERBATIM)
endmacro()

# add unit tests
package_add_test(
  ${PROJECT_NAME}_test ${CMAKE_CURRENT_SOURCE_DIR}/test_zx.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_rational.cpp ${CMAKE_CURRENT_SOURCE_DIR}/test_expression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/test_simplify.cpp)
