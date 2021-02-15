# add pybind11 library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11" "extern/pybind11" EXCLUDE_FROM_ALL)

# add pybind11_json library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11_json" "extern/pybind11_json" EXCLUDE_FROM_ALL)

# add Python interface library
add_library(py${PROJECT_NAME} INTERFACE)

# set include directories
target_include_directories(py${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# link with main project library amd pybind11 libraries
target_link_libraries(py${PROJECT_NAME} INTERFACE ${PROJECT_NAME} pybind11::pybind11)

# add JKQ alias
add_library(JKQ::py${PROJECT_NAME} ALIAS py${PROJECT_NAME})
