if(NOT TARGET gtest OR NOT TARGET gmock)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)

  # store path to googletest in variable
  set(GTEST_PATH "extern/googletest")
  add_subdirectory("${PROJECT_SOURCE_DIR}/${GTEST_PATH}" "${GTEST_PATH}" EXCLUDE_FROM_ALL)
endif()

package_add_test(${PROJECT_NAME}-eval ${PROJECT_NAME}-dd eval_dd_algorithms.cpp)
