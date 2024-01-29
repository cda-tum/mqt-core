# add unit tests
package_add_test(
  ${PROJECT_NAME}-test ${PROJECT_NAME} unittests/test_io.cpp unittests/test_qfr_functionality.cpp
  unittests/test_symbolic.cpp unittests/test_qasm3_parser.cpp)

file(GLOB_RECURSE circuit_optimizer_tests "unittests/circuit_optimizer/*.cpp")
package_add_test(${PROJECT_NAME}-test-circuit-optimizer ${PROJECT_NAME} ${circuit_optimizer_tests})

package_add_test(
  ${PROJECT_NAME}-test-dd
  ${PROJECT_NAME}-dd
  dd/test_complex.cpp
  dd/test_package.cpp
  dd/test_dd_functionality.cpp
  dd/test_dd_noise_functionality.cpp
  algorithms/eval_dynamic_circuits.cpp
  algorithms/test_qft.cpp
  algorithms/test_grover.cpp
  algorithms/test_partial_equivalence.cpp
  algorithms/test_bernsteinvazirani.cpp
  algorithms/test_entanglement.cpp
  algorithms/test_grcs.cpp
  algorithms/test_random_clifford.cpp
  algorithms/test_qpe.cpp
  algorithms/test_wstate.cpp)

package_add_test(${PROJECT_NAME}-test-dd-edge ${PROJECT_NAME}-dd dd/test_edge_functionality.cpp)

file(GLOB_RECURSE zx_tests "zx/*.cpp")
package_add_test(${PROJECT_NAME}-test-zx ${PROJECT_NAME}-zx ${zx_tests})

package_add_test(${PROJECT_NAME}-test-ecc ${PROJECT_NAME}-ecc unittests/test_ecc_functionality.cpp)
target_link_libraries(${PROJECT_NAME}-test-ecc PRIVATE ${PROJECT_NAME}-dd)
