add_executable(${PROJECT_NAME}_app ${CMAKE_CURRENT_SOURCE_DIR}/app.cpp)
target_compile_features(${PROJECT_NAME}_app PUBLIC cxx_std_14)
set_target_properties(${PROJECT_NAME}_app PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(${PROJECT_NAME}_app PUBLIC -mtune=native -march=native -pipe)
target_link_libraries(${PROJECT_NAME}_app PRIVATE ${PROJECT_NAME})

execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_app ${CMAKE_BINARY_DIR}/${PROJECT_NAME}_app)
