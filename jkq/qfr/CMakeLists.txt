# add pybind11 library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11" "extern/pybind11" EXCLUDE_FROM_ALL)

# add pybind11_json library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11_json" "extern/pybind11_json" EXCLUDE_FROM_ALL)

# add Python interface library
add_library(${PROJECT_NAME}_python INTERFACE)

# set include directories
target_include_directories(${PROJECT_NAME}_python INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# link with main project library amd pybind11 libraries
target_link_libraries(${PROJECT_NAME}_python INTERFACE ${PROJECT_NAME} pybind11::pybind11)

# add JKQ alias
add_library(JKQ::${PROJECT_NAME}_python ALIAS ${PROJECT_NAME}_python)

# add pyqfr module
pybind11_add_module(py${PROJECT_NAME} bindings.cpp)
target_link_libraries(py${PROJECT_NAME} PUBLIC ${PROJECT_NAME} JKQ::${PROJECT_NAME}_python pybind11_json)
enable_lto(py${PROJECT_NAME})
