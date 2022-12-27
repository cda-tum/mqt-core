# add MQT::DDPackage library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/dd_package" "extern/dd_package" EXCLUDE_FROM_ALL)

# add MQT::zx library
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/zx" "extern/zx" EXCLUDE_FROM_ALL)

# add nlohmann::json library
set(JSON_BuildTests # cmake-lint: disable=C0103
    OFF
    CACHE INTERNAL "")
set(JSON_MultipleHeaders # cmake-lint: disable=C0103
    OFF
    CACHE INTERNAL "")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern/json" "extern/json" EXCLUDE_FROM_ALL)

# main project library
add_library(
  ${PROJECT_NAME}
  ${PROJECT_SOURCE_DIR}/include/algorithms/BernsteinVazirani.hpp
  ${PROJECT_SOURCE_DIR}/include/algorithms/Entanglement.hpp
  ${PROJECT_SOURCE_DIR}/include/algorithms/GoogleRandomCircuitSampling.hpp
  ${PROJECT_SOURCE_DIR}/include/algorithms/Grover.hpp
  ${PROJECT_SOURCE_DIR}/include/algorithms/QFT.hpp
  ${PROJECT_SOURCE_DIR}/include/algorithms/QPE.hpp
  ${PROJECT_SOURCE_DIR}/include/algorithms/RandomCliffordCircuit.hpp
  ${PROJECT_SOURCE_DIR}/include/CircuitOptimizer.hpp
  ${PROJECT_SOURCE_DIR}/include/Definitions.hpp
  ${PROJECT_SOURCE_DIR}/include/operations/ClassicControlledOperation.hpp
  ${PROJECT_SOURCE_DIR}/include/operations/CompoundOperation.hpp
  ${PROJECT_SOURCE_DIR}/include/operations/Control.hpp
  ${PROJECT_SOURCE_DIR}/include/operations/NonUnitaryOperation.hpp
  ${PROJECT_SOURCE_DIR}/include/operations/Operation.hpp
  ${PROJECT_SOURCE_DIR}/include/operations/StandardOperation.hpp
  ${PROJECT_SOURCE_DIR}/include/operations/SymbolicOperation.hpp
  ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Parser.hpp
  ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Scanner.hpp
  ${PROJECT_SOURCE_DIR}/include/parsers/qasm_parser/Token.hpp
  ${PROJECT_SOURCE_DIR}/include/Permutation.hpp
  ${PROJECT_SOURCE_DIR}/include/QuantumComputation.hpp
  algorithms/BernsteinVazirani.cpp
  algorithms/Entanglement.cpp
  algorithms/GoogleRandomCircuitSampling.cpp
  algorithms/Grover.cpp
  algorithms/QFT.cpp
  algorithms/QPE.cpp
  algorithms/RandomCliffordCircuit.cpp
  CircuitOptimizer.cpp
  operations/NonUnitaryOperation.cpp
  operations/Operation.cpp
  operations/StandardOperation.cpp
  operations/SymbolicOperation.cpp
  parsers/GRCSParser.cpp
  parsers/QASMParser.cpp
  parsers/QCParser.cpp
  parsers/RealParser.cpp
  parsers/TFCParser.cpp
  parsers/qasm_parser/Parser.cpp
  parsers/qasm_parser/Scanner.cpp
  QuantumComputation.cpp)

# set include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include
                                                  ${PROJECT_BINARY_DIR}/include)

target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json)

# add options and warnings to the library
target_link_libraries(${PROJECT_NAME} PUBLIC project_options project_warnings)

# Temporary: The ZX package is not yet clearly separated and some of its definitions are used
# throughout the library. Therefore, the ZX package always needs to be linked into the main QFR
# library.
target_link_libraries(${PROJECT_NAME} PUBLIC MQT::zx)

# add MQT alias
add_library(MQT::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# add DD Package library
add_library(
  ${PROJECT_NAME}_dd INTERFACE ${PROJECT_SOURCE_DIR}/include/dd/FunctionalityConstruction.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/NoiseFunctionality.hpp
  ${PROJECT_SOURCE_DIR}/include/dd/Operations.hpp ${PROJECT_SOURCE_DIR}/include/dd/Simulation.hpp)
target_link_libraries(${PROJECT_NAME}_dd INTERFACE MQT::DDPackage MQT::${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_dd INTERFACE ${PROJECT_SOURCE_DIR}/include/dd
                                                        ${PROJECT_BINARY_DIR}/include/dd)
add_library(MQT::${PROJECT_NAME}_dd ALIAS ${PROJECT_NAME}_dd)

# add ZX package library
add_library(${PROJECT_NAME}_zx ${PROJECT_SOURCE_DIR}/include/zx/FunctionalityConstruction.hpp
                               zx/FunctionalityConstruction.cpp)
target_link_libraries(${PROJECT_NAME}_zx PUBLIC MQT::zx MQT::${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_zx PUBLIC ${PROJECT_SOURCE_DIR}/include/zx
                                                     ${PROJECT_BINARY_DIR}/include/zx)
add_library(MQT::${PROJECT_NAME}_zx ALIAS ${PROJECT_NAME}_zx)

if(GENERATE_POSITION_INDEPENDENT_CODE OR BINDINGS)
  include(CheckPIESupported)
  check_pie_supported()
  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  set_target_properties(${PROJECT_NAME}_dd PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  set_target_properties(${PROJECT_NAME}_zx PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()
