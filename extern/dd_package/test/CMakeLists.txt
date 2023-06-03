if(NOT TARGET gtest OR NOT TARGET gmock)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)
  add_subdirectory("${CMAKE_SOURCE_DIR}/extern/googletest" "extern/googletest" EXCLUDE_FROM_ALL)
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER extern)
endif()

if(NOT TARGET ${PROJECT_NAME}_example)
  add_executable(${PROJECT_NAME}_example main.cpp)
  target_link_libraries(${PROJECT_NAME}_example PRIVATE ${PROJECT_NAME})
  set_target_properties(${PROJECT_NAME}_example PROPERTIES FOLDER tests)

  # include dd_example in test suite
  add_test(NAME ${PROJECT_NAME}_example COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_example)
endif()

# add unit tests
package_add_test(${PROJECT_NAME}_test ${PROJECT_NAME} test_complex.cpp test_package.cpp)
