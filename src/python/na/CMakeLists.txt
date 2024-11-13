if(NOT TARGET na)
  # collect source files
  file(GLOB_RECURSE NA_SOURCES **.cpp)

  # declare the Python module
  pybind11_add_module(
    # Name of the extension
    na
    # Prefer thin LTO if available
    THIN_LTO
    # Optimize the bindings for size
    OPT_SIZE
    # Source code goes here
    ${MQT_CORE_INCLUDE_BUILD_DIR}/python/pybind11.hpp
    ${NA_SOURCES})
  target_link_libraries(na PRIVATE MQT::CoreNA MQT::ProjectOptions MQT::ProjectWarnings)

  # Install directive for scikit-build-core
  install(
    TARGETS na
    DESTINATION .
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()
