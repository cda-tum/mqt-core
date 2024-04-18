#
# This file is part of the MQT CORE library released under the MIT license.
# See README.md or go to https://github.com/cda-tum/mqt-core for more
# information.
#

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-na)
  file(GLOB_RECURSE NA_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/na/**/*.hpp)
  file(GLOB_RECURSE NA_SOURCES *.cpp)

  # add NA package library
  add_library(
    ${MQT_CORE_TARGET_NAME}-na
    ${NA_HEADERS}
    ${NA_SOURCES})

  target_link_libraries(${MQT_CORE_TARGET_NAME}-na PUBLIC MQT::Core)
  target_link_libraries(${MQT_CORE_TARGET_NAME}-na PRIVATE MQT::ProjectOptions
    MQT::ProjectWarnings)

  # add MQT alias
  add_library(MQT::CoreNA ALIAS ${MQT_CORE_TARGET_NAME}-na)
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-na
    PROPERTIES VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    EXPORT_NAME CoreNA)
  set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-na
    PARENT_SCOPE)
endif()
