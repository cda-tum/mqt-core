if(NOT TARGET gtest OR NOT TARGET gmock)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/dd_package/extern/googletest"
                   "extern/dd_package/extern/googletest" EXCLUDE_FROM_ALL)
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER extern)
endif()

# add unit tests
package_add_test(${PROJECT_NAME}_test ${PROJECT_NAME} unittests/test_io.cpp
                 unittests/test_qfr_functionality.cpp)

package_add_test(
  ${PROJECT_NAME}_test_dd
  ${PROJECT_NAME}_dd
  unittests/test_dd_functionality.cpp
  unittests/test_dd_noise_functionality.cpp
  algorithms/eval_dynamic_circuits.cpp
  algorithms/test_qft.cpp
  algorithms/test_grover.cpp
  algorithms/test_bernsteinvazirani.cpp
  algorithms/test_entanglement.cpp
  algorithms/test_grcs.cpp
  algorithms/test_random_clifford.cpp
  algorithms/test_qpe.cpp)

package_add_test(${PROJECT_NAME}_test_zx ${PROJECT_NAME}_zx unittests/test_zx_functionality.cpp
                 unittests/test_symbolic.cpp)

package_add_test(${PROJECT_NAME}_test_ecc ${PROJECT_NAME}_ecc unittests/test_ecc_functionality.cpp)

target_link_libraries(${PROJECT_NAME}_test_ecc PRIVATE ${PROJECT_NAME}_dd)

gtest_discover_tests(${PROJECT_NAME}_test_ecc DISCOVERY_TIMEOUT 1500)
