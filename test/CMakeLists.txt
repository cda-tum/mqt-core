add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")
mark_as_advanced(
		BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
		gmock_build_tests gtest_build_samples gtest_build_tests
		gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)
set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

add_executable(GoogleTest algorithms/test_qft.cpp algorithms/test_grover.cpp)
target_compile_options(GoogleTest PUBLIC -mtune=native -march=native -pipe -Wall -Wextra -Wnon-virtual-dtor -pedantic)
target_link_libraries(GoogleTest PRIVATE QuantumFunctionalityRepresentation gtest gtest_main)
target_compile_features(GoogleTest PUBLIC cxx_std_14)
set_property(TARGET GoogleTest PROPERTY CXX_EXTENSIONS OFF)
set_target_properties(GoogleTest PROPERTIES FOLDER test)

add_executable(QFR_example main.cpp)
target_compile_features(QFR_example PUBLIC cxx_std_14)
set_target_properties(QFR_example PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(QFR_example PUBLIC -mtune=native -march=native -pipe -Wall -Wextra -Wnon-virtual-dtor -pedantic)
target_link_libraries(QFR_example PRIVATE QuantumFunctionalityRepresentation)


file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/circuits
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/qelib1.inc
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
