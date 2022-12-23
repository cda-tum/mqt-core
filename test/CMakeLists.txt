if(NOT TARGET gtest OR NOT TARGET gmock)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/dd_package/extern/googletest"
                   "extern/dd_package/extern/googletest" EXCLUDE_FROM_ALL)
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER extern)
endif()

add_executable(${PROJECT_NAME}_example ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME}_example PRIVATE ${PROJECT_NAME}_dd)

add_custom_command(
  TARGET ${PROJECT_NAME}_example
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/circuits/
          $<TARGET_FILE_DIR:${PROJECT_NAME}_example>/circuits
  COMMENT "Copying circuits for ${PROJECT_NAME}_example"
  VERBATIM)

# add unit tests
package_add_test(
		${PROJECT_NAME}_test ${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_io.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_qfr_functionality.cpp)

package_add_test(
		${PROJECT_NAME}_test_dd
		${PROJECT_NAME}_dd
		${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_dd_functionality.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_dd_noise_functionality.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/algorithms/eval_dynamic_circuits.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_qft.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_grover.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_bernsteinvazirani.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_entanglement.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_grcs.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_random_clifford.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/algorithms/test_qpe.cpp)

package_add_test(
		${PROJECT_NAME}_test_zx ${PROJECT_NAME}_zx
		${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_zx_functionality.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/unittests/test_symbolic.cpp)
