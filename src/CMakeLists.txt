add_library(${PROJECT_NAME} STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/qasm_parser/Token.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/qasm_parser/Scanner.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/qasm_parser/Parser.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/operations/Operation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/operations/StandardOperation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/operations/NonUnitaryOperation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/operations/CompoundOperation.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/operations/ClassicControlledOperation.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/QuantumComputation.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/QFT.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/Grover.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/GoogleRandomCircuitSampling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/Entanglement.cpp

            ${PROJECT_SOURCE_DIR}/include/qasm_parser/Token.hpp
            ${PROJECT_SOURCE_DIR}/include/qasm_parser/Scanner.hpp
            ${PROJECT_SOURCE_DIR}/include/qasm_parser/Parser.hpp

            ${PROJECT_SOURCE_DIR}/include/operations/Operation.hpp
            ${PROJECT_SOURCE_DIR}/include/operations/StandardOperation.hpp
            ${PROJECT_SOURCE_DIR}/include/operations/NonUnitaryOperation.hpp
            ${PROJECT_SOURCE_DIR}/include/operations/CompoundOperation.hpp
            ${PROJECT_SOURCE_DIR}/include/operations/ClassicControlledOperation.hpp

            ${PROJECT_SOURCE_DIR}/include/QuantumComputation.hpp

            ${PROJECT_SOURCE_DIR}/include/algorithms/QFT.hpp
            ${PROJECT_SOURCE_DIR}/include/algorithms/Grover.hpp
            ${PROJECT_SOURCE_DIR}/include/algorithms/GoogleRandomCircuitSampling.hpp
            ${PROJECT_SOURCE_DIR}/include/algorithms/Entanglement.hpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/qasm_parser/ ${PROJECT_SOURCE_DIR}/include/operations/ ${PROJECT_SOURCE_DIR}/include/algorithms/)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/dd_package" "extern/dd_package")
target_link_libraries(${PROJECT_NAME} PUBLIC dd_package_lib)
if (COVERAGE)
	target_compile_options(${PROJECT_NAME} PRIVATE --coverage)
	target_link_libraries(${PROJECT_NAME} PRIVATE --coverage)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
	set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE "/W4" "$<$<CONFIG:RELEASE>:/O2 /GL>")
	target_link_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:/LTCG>")
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -Wall -DNDEBUG -g0 $<$<CONFIG:DEBUG>:-Og -Wextra -Wpedantic -pedantic> $<$<CONFIG:RELEASE>:-O3 -mtune=native -march=native>)
endif()
