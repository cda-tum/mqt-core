add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11" "extern/pybind11" EXCLUDE_FROM_ALL)
add_library(pybind11 ALIAS pybind11::pybind11)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/pybind11_json" "extern/pybind11_json" EXCLUDE_FROM_ALL)
target_compile_options(${PROJECT_NAME} PUBLIC -fvisibility=hidden)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
target_compile_options(nlohmann_json INTERFACE -fvisibility=hidden)
target_compile_options(dd_package PUBLIC -fvisibility=hidden)
set_target_properties(dd_package PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_library(py${PROJECT_NAME} INTERFACE)
add_library(JKQ::py${PROJECT_NAME} ALIAS py${PROJECT_NAME})
target_include_directories(py${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(py${PROJECT_NAME} INTERFACE ${PROJECT_NAME} pybind11::pybind11)
target_compile_options(py${PROJECT_NAME} INTERFACE -fvisibility=hidden)

