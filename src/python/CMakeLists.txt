if(NOT TARGET _core)
  # collect source files
  file(GLOB_RECURSE IR_PYTHON_SOURCES ir/*.cpp)

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
    ${IR_PYTHON_SOURCES})
  target_link_libraries(_core PRIVATE MQT::CoreIR MQT::ProjectOptions MQT::ProjectWarnings)

  # Install directive for scikit-build-core
  install(
    TARGETS _core
    DESTINATION .
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()
