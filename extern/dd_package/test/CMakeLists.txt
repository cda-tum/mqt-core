if(NOT TARGET gtest OR NOT TARGET gmock)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest" EXCLUDE_FROM_ALL)
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER extern)
endif()

if(NOT TARGET benchmark::benchmark)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
  set(BENCHMARK_ENABLE_TESTING
      OFF
      CACHE BOOL "Skip google benchmark self tests")
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/benchmark" "extern/benchmark" EXCLUDE_FROM_ALL)
  set_target_properties(benchmark benchmark_main PROPERTIES FOLDER extern)
endif()

add_executable(${PROJECT_NAME}_example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME}_example PRIVATE ${PROJECT_NAME})
set_target_properties(${PROJECT_NAME}_example PROPERTIES FOLDER tests)

# add unit tests
package_add_test(${PROJECT_NAME}_test ${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/test_complex.cpp
                 ${CMAKE_CURRENT_SOURCE_DIR}/test_package.cpp)

# include dd_example in test suite
add_test(NAME ${PROJECT_NAME}_example COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_example)

# add benchmark executable
add_executable(${PROJECT_NAME}_bench EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/bench_package.cpp)
# link the Google benchmark infrastructure and a default main function to the benchmark executable.
target_link_libraries(${PROJECT_NAME}_bench PRIVATE ${PROJECT_NAME} benchmark::benchmark_main
                                                    Threads::Threads)
set_target_properties(${PROJECT_NAME}_bench PROPERTIES FOLDER tests)
