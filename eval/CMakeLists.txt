if(NOT TARGET gtest OR NOT TARGET gmock)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest" EXCLUDE_FROM_ALL)
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER extern)
endif()

add_executable(${PROJECT_NAME}-dd-eval eval_dd_package.cpp)
target_link_libraries(${PROJECT_NAME}-dd-eval ${PROJECT_NAME}-dd gtest)
gtest_discover_tests(${PROJECT_NAME}-dd-eval)
