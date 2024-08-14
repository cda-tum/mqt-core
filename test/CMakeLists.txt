# add various subdirectories containing tests
add_subdirectory(algorithms)
add_subdirectory(circuit_optimizer)
add_subdirectory(datastructures)
add_subdirectory(dd)
add_subdirectory(ir)
add_subdirectory(na)
add_subdirectory(zx)

# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
