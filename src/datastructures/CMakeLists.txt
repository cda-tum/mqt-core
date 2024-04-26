#
# This file is part of the MQT CORE library released under the MIT license. See README.md or go to
# https://github.com/cda-tum/mqt-core for more information.
#

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-ds)
  file(GLOB_RECURSE DS_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/datastructures/**.hpp)
  file(GLOB_RECURSE DS_SOURCES **.cpp)

  # add NA package library
  add_library(${MQT_CORE_TARGET_NAME}-ds ${DS_HEADERS} ${DS_SOURCES})

  target_link_libraries(${MQT_CORE_TARGET_NAME}-ds PUBLIC MQT::Core)
  target_link_libraries(${MQT_CORE_TARGET_NAME}-ds PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  # add MQT alias
  add_library(MQT::CoreDS ALIAS ${MQT_CORE_TARGET_NAME}-ds)
  set_target_properties(
    ${MQT_CORE_TARGET_NAME}-ds
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreDS)
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-ds
      PARENT_SCOPE)
endif()
