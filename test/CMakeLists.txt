if(NOT TARGET gtest OR NOT TARGET gmock)
  # Prevent overriding the parent project's compiler/linker settings on Windows
  set(gtest_force_shared_crt # cmake-lint: disable=C0103
      ON
      CACHE BOOL "" FORCE)
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest" EXCLUDE_FROM_ALL)
  set_target_properties(gtest gtest_main gmock gmock_main PROPERTIES FOLDER extern)
endif()

# add unit tests
package_add_test(${PROJECT_NAME}-test ${PROJECT_NAME} unittests/test_io.cpp
                 unittests/test_qfr_functionality.cpp unittests/test_symbolic.cpp)

package_add_test(
  ${PROJECT_NAME}-test-dd
  ${PROJECT_NAME}-dd
  dd/test_package.cpp
  dd/test_dd_functionality.cpp
  dd/test_dd_noise_functionality.cpp
  algorithms/eval_dynamic_circuits.cpp
  algorithms/test_qft.cpp
  algorithms/test_grover.cpp
  algorithms/test_bernsteinvazirani.cpp
  algorithms/test_entanglement.cpp
  algorithms/test_grcs.cpp
  algorithms/test_random_clifford.cpp
  algorithms/test_qpe.cpp
  algorithms/test_wstate.cpp)

package_add_test(${PROJECT_NAME}-test-dd-edge ${PROJECT_NAME}-dd dd/test_edge_functionality.cpp)

package_add_test(
  ${PROJECT_NAME}-test-zx
  ${PROJECT_NAME}-zx
  zx/test_zx.cpp
  zx/test_rational.cpp
  zx/test_expression.cpp
  zx/test_simplify.cpp
  zx/test_zx_functionality.cpp)

package_add_test(${PROJECT_NAME}-test-ecc ${PROJECT_NAME}-ecc unittests/test_ecc_functionality.cpp)
target_link_libraries(${PROJECT_NAME}-test-ecc PRIVATE ${PROJECT_NAME}-dd)

package_add_test(${PROJECT_NAME}-eval-id-removal ${PROJECT_NAME}-dd dd/eval_identity_removal.cpp)
