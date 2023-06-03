# add pybind11 library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11" "extern/pybind11" EXCLUDE_FROM_ALL)

if(NOT TARGET pybind11_json)
  # add pybind11_json library
  add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11_json" "extern/pybind11_json"
                   EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET ${PROJECT_NAME}_python)
  # add Python interface library
  add_library(${PROJECT_NAME}_python INTERFACE)

  # set include directories
  target_include_directories(${PROJECT_NAME}_python INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

  # link with main project library and pybind11 libraries
  target_link_libraries(${PROJECT_NAME}_python INTERFACE ${PROJECT_NAME} pybind11::pybind11
                                                         pybind11_json)

  # add MQT alias
  add_library(MQT::${PROJECT_NAME}_python ALIAS ${PROJECT_NAME}_python)
endif()
