if(NOT SKBUILD)
  # Manually detect the installed pybind11 package and import it into CMake.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE pybind11_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
endif()

# add pybind11 library
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
  # Name of the extension
  _core
  # Target the stable ABI for Python 3.12+, which reduces the number of binary wheels that must be
  # built. This does nothing on older Python versions
  WITH_SOABI
  # Source code goes here
  ${MQT_CORE_INCLUDE_BUILD_DIR}/python/pybind11.hpp
  module.cpp
  register_operations.cpp
  register_permutation.cpp
  register_symbolic.cpp
  register_quantum_computation.cpp
  operations/register_optype.cpp
  operations/register_control.cpp
  operations/register_operation.cpp
  operations/register_standard_operation.cpp
  operations/register_classic_controlled_operation.cpp
  operations/register_compound_operation.cpp
  operations/register_non_unitary_operation.cpp
  operations/register_symbolic_operation.cpp
  symbolic/register_variable.cpp
  symbolic/register_term.cpp
  symbolic/register_expression.cpp)
target_link_libraries(_core PRIVATE MQT::Core)

# Install directive for scikit-build-core
install(
  TARGETS _core
  DESTINATION .
  COMPONENT MQTCorePythonModule)
