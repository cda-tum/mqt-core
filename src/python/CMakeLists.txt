if(NOT TARGET _core)
  pybind11_add_module(
    # Name of the extension
    _core
    # Prefer thin LTO if available
    THIN_LTO
    # Optimize the bindings for size
    OPT_SIZE
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
  target_link_libraries(_core PRIVATE MQT::Core MQT::ProjectOptions MQT::ProjectWarnings)

  # Install directive for scikit-build-core
  install(
    TARGETS _core
    DESTINATION .
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()
